import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")


def plot_equity_curve(equity_curve: pd.Series, 
                     benchmark: Optional[pd.Series] = None,
                     title: str = "Portfolio Performance",
                     figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None):
    """
    Plot equity curve with optional benchmark comparison.
    
    Args:
        equity_curve: Series with portfolio equity values and datetime index
        benchmark: Optional benchmark series with same index
        title: Plot title
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize series to start at 100
    equity_norm = equity_curve / equity_curve.iloc[0] * 100
    ax.plot(equity_norm.index, equity_norm, label='Portfolio', linewidth=2)
    
    if benchmark is not None:
        # Normalize benchmark to start at 100
        benchmark_norm = benchmark / benchmark.iloc[0] * 100
        ax.plot(benchmark_norm.index, benchmark_norm, label='Benchmark', linewidth=2, alpha=0.7)
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add labels and legend
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Value (Starting = 100)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_drawdowns(equity_curve: pd.Series,
                  title: str = "Portfolio Drawdowns",
                  figsize: Tuple[int, int] = (12, 6),
                  highlight_threshold: float = -0.1,
                  save_path: Optional[str] = None):
    """
    Plot portfolio drawdowns over time.
    
    Args:
        equity_curve: Series with portfolio equity values and datetime index
        title: Plot title
        figsize: Figure size as tuple (width, height)
        highlight_threshold: Threshold to highlight significant drawdowns
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdowns
    drawdowns = (equity_curve - running_max) / running_max
    
    # Plot drawdowns
    ax.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax.plot(drawdowns.index, drawdowns, color='darkred', linewidth=1)
    
    # Highlight significant drawdowns
    significant_drawdowns = drawdowns[drawdowns <= highlight_threshold]
    if len(significant_drawdowns) > 0:
        ax.scatter(significant_drawdowns.index, significant_drawdowns, 
                  color='darkred', s=50, zorder=5,
                  label=f'Drawdowns ≤ {highlight_threshold:.0%}')
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add labels and legend
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if there are significant drawdowns
    if len(significant_drawdowns) > 0:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_rolling_metrics(returns: pd.Series,
                        window: int = 60,
                        metrics: List[str] = ['return', 'volatility', 'sharpe'],
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None):
    """
    Plot rolling performance metrics over time.
    
    Args:
        returns: Series with daily returns and datetime index
        window: Rolling window size in days
        metrics: List of metrics to plot
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    # Validate metrics
    valid_metrics = ['return', 'volatility', 'sharpe', 'sortino', 'calmar']
    metrics = [m for m in metrics if m in valid_metrics]
    
    if not metrics:
        raise ValueError(f"No valid metrics specified. Choose from: {valid_metrics}")
    
    # Number of subplots
    n_plots = len(metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    # Convert to single axes if only one metric
    if n_plots == 1:
        axes = [axes]
    
    # Calculate rolling metrics
    rolling_return = returns.rolling(window=window).mean() * 252  # Annualized
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Calculate rolling Sharpe ratio (assuming 0 risk-free rate for simplicity)
    rolling_sharpe = rolling_return / rolling_vol
    
    # Calculate rolling Sortino ratio (downside deviation)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    rolling_downside_vol = downside_returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sortino = rolling_return / rolling_downside_vol
    
    # Calculate rolling maximum drawdown for Calmar ratio
    equity_curve = (1 + returns).cumprod()
    rolling_max_dd = pd.Series(index=returns.index)
    
    for i in range(window, len(equity_curve)):
        window_equity = equity_curve.iloc[i-window:i]
        window_peak = window_equity.expanding().max()
        window_dd = (window_equity - window_peak) / window_peak
        rolling_max_dd.iloc[i] = window_dd.min()
    
    # Calculate rolling Calmar ratio
    rolling_calmar = rolling_return / abs(rolling_max_dd)
    
    # Colors for different metrics
    colors = {
        'return': 'green',
        'volatility': 'red',
        'sharpe': 'blue',
        'sortino': 'purple',
        'calmar': 'orange'
    }
    
    # Plotting each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric == 'return':
            ax.plot(rolling_return.index, rolling_return, 
                   color=colors[metric], linewidth=2, 
                   label=f'{window}-day Rolling Annualized Return')
            ax.set_ylabel('Annualized Return', fontsize=10)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
        elif metric == 'volatility':
            ax.plot(rolling_vol.index, rolling_vol, 
                   color=colors[metric], linewidth=2, 
                   label=f'{window}-day Rolling Annualized Volatility')
            ax.set_ylabel('Annualized Volatility', fontsize=10)
            
        elif metric == 'sharpe':
            ax.plot(rolling_sharpe.index, rolling_sharpe, 
                   color=colors[metric], linewidth=2, 
                   label=f'{window}-day Rolling Sharpe Ratio')
            ax.set_ylabel('Sharpe Ratio', fontsize=10)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
            
        elif metric == 'sortino':
            ax.plot(rolling_sortino.index, rolling_sortino, 
                   color=colors[metric], linewidth=2, 
                   label=f'{window}-day Rolling Sortino Ratio')
            ax.set_ylabel('Sortino Ratio', fontsize=10)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
            
        elif metric == 'calmar':
            ax.plot(rolling_calmar.index, rolling_calmar, 
                   color=colors[metric], linewidth=2, 
                   label=f'{window}-day Rolling Calmar Ratio')
            ax.set_ylabel('Calmar Ratio', fontsize=10)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        
        # Add legend
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Format x-axis dates on the bottom plot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    axes[-1].set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_monthly_returns_heatmap(returns: pd.Series,
                               title: str = "Monthly Returns Heatmap",
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None):
    """
    Plot a heatmap of monthly returns.
    
    Args:
        returns: Series with daily returns and datetime index
        title: Plot title
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    # Resample to get monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table for the heatmap
    returns_table = pd.pivot_table(
        monthly_returns.reset_index(),
        values=monthly_returns.name if monthly_returns.name else 0,
        index=monthly_returns.index.month,
        columns=monthly_returns.index.year
    )
    
    # Replace month numbers with month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    returns_table.index = [month_names[i-1] for i in returns_table.index]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    heatmap = sns.heatmap(returns_table * 100, 
                        annot=True, 
                        fmt=".1f", 
                        cmap="RdYlGn",
                        center=0,
                        cbar_kws={'label': 'Return (%)'},
                        linewidths=0.5,
                        ax=ax)
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Month', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_strategy_correlations(returns_dict: Dict[str, pd.Series],
                              title: str = "Strategy Correlations",
                              figsize: Tuple[int, int] = (10, 8),
                              save_path: Optional[str] = None):
    """
    Plot correlation matrix of strategy returns.
    
    Args:
        returns_dict: Dictionary mapping strategy names to return series
        title: Plot title
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    # Create DataFrame from returns dictionary
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    heatmap = sns.heatmap(corr_matrix, 
                        annot=True, 
                        fmt=".2f", 
                        cmap="coolwarm",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        mask=mask,
                        cbar_kws={'label': 'Correlation'},
                        linewidths=0.5,
                        ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_return_distribution(returns: pd.Series,
                           title: str = "Return Distribution",
                           bins: int = 50,
                           figsize: Tuple[int, int] = (12, 6),
                           var_percentile: float = 0.05,
                           save_path: Optional[str] = None):
    """
    Plot distribution of returns with VaR and normal distribution overlay.
    
    Args:
        returns: Series with daily returns
        title: Plot title
        bins: Number of bins for histogram
        figsize: Figure size as tuple (width, height)
        var_percentile: Percentile for Value at Risk (default: 5%)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate stats
    mean_return = returns.mean()
    std_return = returns.std()
    skew = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Calculate VaR
    var = np.percentile(returns, var_percentile * 100)
    
    # Plot histogram
    sns.histplot(returns, bins=bins, kde=True, ax=ax)
    
    # Plot normal distribution for comparison
    x = np.linspace(returns.min(), returns.max(), 1000)
    y = stats.norm.pdf(x, mean_return, std_return)
    ax.plot(x, y * len(returns) * (returns.max() - returns.min()) / bins, 
           'r-', linewidth=2, label='Normal Distribution')
    
    # Add VaR line
    ax.axvline(x=var, color='darkred', linestyle='--', 
              linewidth=2, label=f'{var_percentile*100:.0f}% VaR: {var:.2%}')
    
    # Add statistics as text box
    stats_text = (f"Mean: {mean_return:.4f}\n"
                 f"Std Dev: {std_return:.4f}\n"
                 f"Skewness: {skew:.4f}\n"
                 f"Kurtosis: {kurtosis:.4f}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_strategy_allocations(allocation_history: List[Dict],
                            strategies: List[str],
                            title: str = "Strategy Allocations Over Time",
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None):
    """
    Plot strategy allocations over time.
    
    Args:
        allocation_history: List of allocation dictionaries with dates
        strategies: List of strategy names
        title: Plot title
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    # Create DataFrame from allocation history
    allocation_df = pd.DataFrame([
        {
            'date': entry['date'],
            **{f"{s}_allocation": entry.get(f"{s}_allocation", 0) for s in strategies}
        }
        for entry in allocation_history
    ])
    
    # Set date as index
    allocation_df.set_index('date', inplace=True)
    allocation_df = allocation_df.sort_index()
    
    # Extract allocation columns
    allocation_columns = [f"{s}_allocation" for s in strategies]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot allocations
    ax.stackplot(allocation_df.index, 
                [allocation_df[col] for col in allocation_columns],
                labels=strategies, alpha=0.8)
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add labels and legend
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Allocation (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_regime_analysis(returns: pd.Series,
                       regimes: pd.Series,
                       title: str = "Performance by Market Regime",
                       figsize: Tuple[int, int] = (15, 10),
                       save_path: Optional[str] = None):
    """
    Plot performance metrics by market regime.
    
    Args:
        returns: Series with daily returns and datetime index
        regimes: Series with regime labels and same index as returns
        title: Plot title
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    # Validate input
    if not returns.index.equals(regimes.index):
        raise ValueError("Returns and regimes must have the same index")
    
    # Create a DataFrame with returns and regimes
    df = pd.DataFrame({'returns': returns, 'regime': regimes})
    
    # Get unique regimes
    unique_regimes = df['regime'].unique()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: Cumulative returns by regime
    ax = axes[0]
    
    # Calculate cumulative returns for each regime
    for regime in unique_regimes:
        regime_returns = df[df['regime'] == regime]['returns']
        if len(regime_returns) > 0:
            cum_returns = (1 + regime_returns).cumprod()
            ax.plot(cum_returns.index, cum_returns, label=regime)
    
    ax.set_title('Cumulative Returns by Regime', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of daily returns by regime
    ax = axes[1]
    
    sns.boxplot(x='regime', y='returns', data=df, ax=ax)
    ax.set_title('Distribution of Daily Returns by Regime', fontsize=12)
    ax.set_ylabel('Daily Return', fontsize=10)
    ax.set_xlabel('Regime', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Annualized performance metrics by regime
    ax = axes[2]
    
    # Calculate metrics by regime
    regime_metrics = []
    for regime in unique_regimes:
        regime_returns = df[df['regime'] == regime]['returns']
        if len(regime_returns) > 0:
            annualized_return = regime_returns.mean() * 252
            annualized_vol = regime_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            regime_metrics.append({
                'Regime': regime,
                'Annualized Return': annualized_return,
                'Annualized Volatility': annualized_vol,
                'Sharpe Ratio': sharpe_ratio
            })
    
    metrics_df = pd.DataFrame(regime_metrics)
    metrics_df.set_index('Regime', inplace=True)
    
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Annualized Metrics by Regime', fontsize=12)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Regime duration analysis
    ax = axes[3]
    
    # Calculate regime durations
    regime_changes = df['regime'] != df['regime'].shift(1)
    regime_start_indices = list(df.index[regime_changes]) + [df.index[-1]]
    
    regime_periods = []
    for i in range(len(regime_start_indices) - 1):
        start_date = regime_start_indices[i]
        end_date = regime_start_indices[i+1]
        regime = df.loc[start_date, 'regime']
        duration_days = (end_date - start_date).days
        
        regime_periods.append({
            'Regime': regime,
            'Start': start_date,
            'End': end_date,
            'Duration (days)': duration_days
        })
    
    durations_df = pd.DataFrame(regime_periods)
    
    # Plot average duration by regime
    avg_durations = durations_df.groupby('Regime')['Duration (days)'].mean()
    avg_durations.plot(kind='bar', ax=ax)
    
    ax.set_title('Average Regime Duration', fontsize=12)
    ax.set_ylabel('Duration (days)', fontsize=10)
    ax.set_xlabel('Regime', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def create_performance_dashboard(metrics: Dict,
                               returns: pd.Series,
                               equity_curve: pd.Series,
                               benchmark: Optional[pd.Series] = None,
                               allocation_history: Optional[List[Dict]] = None,
                               strategies: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (15, 20),
                               save_path: Optional[str] = None):
    """
    Create a comprehensive performance dashboard with multiple plots.
    
    Args:
        metrics: Dictionary with performance metrics
        returns: Series with daily returns and datetime index
        equity_curve: Series with portfolio equity values and same index as returns
        benchmark: Optional benchmark series with same index
        allocation_history: Optional list of allocation dictionaries with dates
        strategies: Optional list of strategy names
        figsize: Figure size as tuple (width, height)
        save_path: Optional path to save the figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = fig.add_gridspec(5, 2)
    
    # Plot 1: Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    equity_norm = equity_curve / equity_curve.iloc[0] * 100
    ax1.plot(equity_norm.index, equity_norm, label='Portfolio', linewidth=2)
    
    if benchmark is not None:
        benchmark_norm = benchmark / benchmark.iloc[0] * 100
        ax1.plot(benchmark_norm.index, benchmark_norm, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax1.set_title('Portfolio Performance', fontsize=14)
    ax1.set_ylabel('Value (Starting = 100)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drawdowns
    ax2 = fig.add_subplot(gs[1, :])
    running_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - running_max) / running_max
    ax2.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax2.plot(drawdowns.index, drawdowns, color='darkred', linewidth=1)
    ax2.set_title('Portfolio Drawdowns', fontsize=14)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling metrics (Sharpe, Sortino)
    ax3 = fig.add_subplot(gs[2, 0])
    window = 60  # 60-day rolling window
    
    # Calculate rolling Sharpe and Sortino ratios
    rolling_return = returns.rolling(window=window).mean() * 252
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    rolling_downside_vol = downside_returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sortino = rolling_return / rolling_downside_vol
    
    ax3.plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe', color='blue')
    ax3.plot(rolling_sortino.index, rolling_sortino, label='Rolling Sortino', color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax3.set_title(f'{window}-day Rolling Risk-Adjusted Metrics', fontsize=14)
    ax3.set_ylabel('Ratio Value', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Return distribution
    ax4 = fig.add_subplot(gs[2, 1])
    sns.histplot(returns, bins=50, kde=True, ax=ax4)
    
    # Get VaR
    var_95 = metrics.get('var_95', np.percentile(returns, 5))
    
    # Add VaR line
    ax4.axvline(x=var_95, color='darkred', linestyle='--', 
               linewidth=2, label=f'95% VaR: {var_95:.2%}')
    
    # Add statistics as text box
    mean_return = returns.mean()
    std_return = returns.std()
    skew = returns.skew()
    kurtosis = returns.kurtosis()
    
    stats_text = (f"Mean: {mean_return:.4f}\n"
                 f"Std Dev: {std_return:.4f}\n"
                 f"Skewness: {skew:.4f}\n"
                 f"Kurtosis: {kurtosis:.4f}")
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax4.set_title('Return Distribution', fontsize=14)
    ax4.set_xlabel('Daily Return', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Monthly returns heatmap
    ax5 = fig.add_subplot(gs[3, :])
    
    # Resample to get monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table for the heatmap
    returns_table = pd.pivot_table(
        monthly_returns.reset_index(),
        values=monthly_returns.name if monthly_returns.name else 0,
        index=monthly_returns.index.month,
        columns=monthly_returns.index.year
    )
    
    # Replace month numbers with month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    returns_table.index = [month_names[i-1] for i in returns_table.index]
    
    # Plot heatmap
    sns.heatmap(returns_table * 100, 
               annot=True, 
               fmt=".1f", 
               cmap="RdYlGn",
               center=0,
               cbar_kws={'label': 'Return (%)'},
               linewidths=0.5,
               ax=ax5)
    
    ax5.set_title('Monthly Returns (%)', fontsize=14)
    ax5.set_ylabel('Month', fontsize=12)
    ax5.set_xlabel('Year', fontsize=12)
    
    # Plot 6: Strategy allocations or metrics table
    ax6 = fig.add_subplot(gs[4, :])
    
    if allocation_history and strategies:
        # Create DataFrame from allocation history
        allocation_df = pd.DataFrame([
            {
                'date': entry['date'],
                **{f"{s}_allocation": entry.get(f"{s}_allocation", 0) for s in strategies}
            }
            for entry in allocation_history
        ])
        
        # Set date as index
        allocation_df.set_index('date', inplace=True)
        allocation_df = allocation_df.sort_index()
        
        # Extract allocation columns
        allocation_columns = [f"{s}_allocation" for s in strategies]
        
        # Plot allocations
        ax6.stackplot(allocation_df.index, 
                     [allocation_df[col] for col in allocation_columns],
                     labels=strategies, alpha=0.8)
        
        ax6.set_title('Strategy Allocations Over Time', fontsize=14)
        ax6.set_ylabel('Allocation (%)', fontsize=12)
        ax6.set_xlabel('Date', fontsize=12)
        ax6.legend(loc='upper left')
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)
    else:
        # Create a metrics table
        metrics_list = [
            ('Total Return', f"{metrics.get('total_return', 0) * 100:.2f}%"),
            ('Annualized Return', f"{metrics.get('annualized_return', 0) * 100:.2f}%"),
            ('Volatility', f"{metrics.get('volatility', 0) * 100:.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0) * 100:.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2f}%"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Max Consecutive Wins', f"{metrics.get('max_consecutive_wins', 0)}"),
            ('Max Consecutive Losses', f"{metrics.get('max_consecutive_losses', 0)}")
        ]
        
        # Create a table-like visualization
        table_data = [list(item) for item in metrics_list]
        
        ax6.axis('tight')
        ax6.axis('off')
        ax6.table(cellText=table_data, 
                 colLabels=['Metric', 'Value'], 
                 loc='center',
                 cellLoc='center',
                 colWidths=[0.5, 0.3])
        
        ax6.set_title('Performance Metrics Summary', fontsize=14)
    
    # Format dates on x-axes
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set overall title with main metrics
    total_return = metrics.get('total_return', 0) * 100
    sharpe = metrics.get('sharpe_ratio', 0)
    max_dd = metrics.get('max_drawdown', 0) * 100
    
    fig.suptitle(f'Performance Dashboard\nReturn: {total_return:.2f}% | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2f}%', 
                fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 