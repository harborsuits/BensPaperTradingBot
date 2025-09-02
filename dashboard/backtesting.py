"""
Enhanced backtesting visualization and metrics for the BensBot Trading Dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from dashboard.theme import COLORS

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from a returns series
    
    Args:
        returns_series: Series of returns (not cumulative)
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert annual risk-free rate to match the returns series frequency
    # This is a simplification - would need adjustment based on actual data frequency
    rf_daily = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate metrics
    total_return = (1 + returns_series).prod() - 1
    cagr = (1 + total_return) ** (252 / len(returns_series)) - 1
    
    # Volatility (annualized)
    volatility = returns_series.std() * np.sqrt(252)
    
    # Sharpe Ratio
    excess_returns = returns_series - rf_daily
    sharpe_ratio = (excess_returns.mean() / returns_series.std()) * np.sqrt(252)
    
    # Sortino Ratio - only considering downside deviation
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) > 0:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = 0 if downside_deviation == 0 else (returns_series.mean() - rf_daily) * np.sqrt(252) / downside_deviation
    else:
        sortino_ratio = np.inf  # No downside returns
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min()
    
    # Calmar Ratio
    calmar_ratio = 0 if max_drawdown == 0 else cagr / abs(max_drawdown)
    
    # Win/Loss Metrics
    wins = len(returns_series[returns_series > 0])
    losses = len(returns_series[returns_series < 0])
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # Average Win/Loss
    avg_win = returns_series[returns_series > 0].mean() if wins > 0 else 0
    avg_loss = returns_series[returns_series < 0].mean() if losses > 0 else 0
    
    # Profit Factor
    profit_factor = (returns_series[returns_series > 0].sum() / 
                    abs(returns_series[returns_series < 0].sum())) if losses > 0 else np.inf
    
    # Recovery Factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    # Maximum Consecutive Wins/Losses
    sign_changes = (returns_series > 0).astype(int).diff().fillna(0)
    win_streak = 0
    loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for r in returns_series:
        if r > 0:
            current_win_streak += 1
            current_loss_streak = 0
        elif r < 0:
            current_loss_streak += 1
            current_win_streak = 0
        else:  # r == 0
            # No change, continue current streaks
            pass
        
        win_streak = max(win_streak, current_win_streak)
        loss_streak = max(loss_streak, current_loss_streak)
    
    return {
        "total_return": total_return * 100,  # as percentage
        "cagr": cagr * 100,  # as percentage
        "volatility": volatility * 100,  # as percentage
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown * 100,  # as percentage
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate * 100,  # as percentage
        "avg_win": avg_win * 100,  # as percentage
        "avg_loss": avg_loss * 100,  # as percentage
        "profit_factor": profit_factor,
        "recovery_factor": recovery_factor,
        "max_win_streak": win_streak,
        "max_loss_streak": loss_streak
    }

def create_equity_curve_chart(returns_series: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> go.Figure:
    """
    Create an equity curve chart with drawdown visualization
    
    Args:
        returns_series: Series of returns (not cumulative)
        benchmark_returns: Optional benchmark returns for comparison
        
    Returns:
        Plotly figure with equity curve and drawdown
    """
    # Calculate cumulative returns
    equity_curve = (1 + returns_series).cumprod()
    
    # Calculate drawdown
    drawdowns = (equity_curve / equity_curve.cummax()) - 1
    
    # Create subplots: equity curve on top, drawdown on bottom
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Equity Curve", "Drawdown")
    )
    
    # Add equity curve trace
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index, 
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color=COLORS['primary'], width=2)
        ),
        row=1, col=1
    )
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        benchmark_curve = (1 + benchmark_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index, 
                y=benchmark_curve.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='rgba(100, 100, 100, 0.8)', width=1.5, dash='dot')
            ),
            row=1, col=1
        )
    
    # Add drawdown trace
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index, 
            y=drawdowns.values * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color=COLORS['danger'], width=1.5),
            fill='tozeroy',
            fillcolor=f"rgba({int(COLORS['danger'][1:3], 16)}, {int(COLORS['danger'][3:5], 16)}, {int(COLORS['danger'][5:7], 16)}, 0.1)"
        ),
        row=2, col=1
    )
    
    # Add horizontal line at 0 for drawdown
    fig.add_shape(
        type="line",
        x0=drawdowns.index.min(),
        x1=drawdowns.index.max(),
        y0=0,
        y1=0,
        line=dict(color="grey", width=1, dash="dot"),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            title="Value",
            tickformat='.2f',
        ),
        yaxis2=dict(
            title="Drawdown (%)",
            tickformat='.1f',
            ticksuffix='%',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig

def create_rolling_returns_chart(returns_series: pd.Series, window_sizes: List[int] = [21, 63, 252]) -> go.Figure:
    """
    Create a chart of rolling returns over different time windows
    
    Args:
        returns_series: Series of returns (not cumulative)
        window_sizes: List of window sizes in days
        
    Returns:
        Plotly figure with rolling returns
    """
    fig = go.Figure()
    
    for window in window_sizes:
        # Calculate rolling returns
        rolling_returns = ((1 + returns_series).rolling(window).prod() - 1) * 100
        
        # Determine what to call this window
        window_name = f"{window} days"
        if window == 21:
            window_name = "Monthly"
        elif window == 63:
            window_name = "Quarterly"
        elif window == 252:
            window_name = "Annual"
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns.values,
                mode='lines',
                name=window_name,
            )
        )
    
    # Add horizontal line at 0
    fig.add_shape(
        type="line",
        x0=returns_series.index.min(),
        x1=returns_series.index.max(),
        y0=0,
        y1=0,
        line=dict(color="grey", width=1, dash="dot")
    )
    
    # Update layout
    fig.update_layout(
        title="Rolling Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        yaxis=dict(
            tickformat='.1f',
            ticksuffix='%',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    return fig

def create_monthly_returns_heatmap(returns_series: pd.Series) -> go.Figure:
    """
    Create a heatmap of monthly returns
    
    Args:
        returns_series: Series of returns (not cumulative)
        
    Returns:
        Plotly heatmap figure
    """
    # Resample to get monthly returns
    monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table of returns by year and month
    monthly_returns.index = monthly_returns.index.to_period('M')
    monthly_returns = monthly_returns.reset_index()
    monthly_returns['Year'] = monthly_returns['index'].dt.year
    monthly_returns['Month'] = monthly_returns['index'].dt.month
    monthly_returns = monthly_returns.pivot_table(
        values=0,  # Series values
        index='Year',
        columns='Month'
    )
    
    # Replace month numbers with names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns.columns = [month_names[i-1] for i in monthly_returns.columns]
    
    # Create a custom color scale - positive returns in green, negative in red
    colorscale = [
        [0, f"rgba({int(COLORS['danger'][1:3], 16)}, {int(COLORS['danger'][3:5], 16)}, {int(COLORS['danger'][5:7], 16)}, 0.8)"],
        [0.5, "rgba(255, 255, 255, 0.8)"],
        [1, f"rgba({int(COLORS['success'][1:3], 16)}, {int(COLORS['success'][3:5], 16)}, {int(COLORS['success'][5:7], 16)}, 0.8)"]
    ]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns.values * 100,  # Convert to percentage
        x=monthly_returns.columns,
        y=monthly_returns.index,
        colorscale=colorscale,
        zmid=0,  # Center the color scale at 0
        text=[[f"{val*100:.1f}%" for val in row] for row in monthly_returns.values],
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    # Update layout
    fig.update_layout(
        title="Monthly Returns (%)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_nticks=12,
        yaxis_nticks=len(monthly_returns.index),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    return fig

def create_distribution_chart(returns_series: pd.Series) -> go.Figure:
    """
    Create a distribution chart of returns
    
    Args:
        returns_series: Series of returns (not cumulative)
        
    Returns:
        Plotly figure with return distribution
    """
    # Convert to percentage
    returns_pct = returns_series * 100
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns_pct,
        histnorm='probability density',
        name='Returns',
        marker_color=COLORS['primary'],
        opacity=0.75,
        nbinsx=30
    ))
    
    # Add normal distribution curve
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    y = 1/(returns_pct.std() * np.sqrt(2 * np.pi)) * np.exp(-(x - returns_pct.mean())**2 / (2 * returns_pct.std()**2))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color=COLORS['secondary'], width=2, dash='dash')
    ))
    
    # Add a vertical line at 0
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="grey", width=1, dash="dot")
    )
    
    # Update layout
    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Probability Density",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    return fig

def generate_customizable_backtest_metrics(
    strategy_returns: pd.Series, 
    benchmark_returns: Optional[pd.Series] = None,
    custom_metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate complete backtest metrics and charts with customization options
    
    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Optional benchmark returns for comparison
        custom_metrics: List of specific metrics to include (or None for all)
        
    Returns:
        Dictionary with metrics and chart figures
    """
    # Calculate metrics
    metrics = calculate_performance_metrics(strategy_returns)
    
    if benchmark_returns is not None:
        benchmark_metrics = calculate_performance_metrics(benchmark_returns)
        
        # Calculate additional comparison metrics
        metrics['alpha'] = metrics['cagr'] - benchmark_metrics['cagr']
        
        # Beta calculation (covariance / variance)
        beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        metrics['beta'] = beta
        
        # Information ratio
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
        metrics['information_ratio'] = (metrics['cagr'] - benchmark_metrics['cagr']) / tracking_error
        
        # Add benchmark metrics with prefix
        for key, value in benchmark_metrics.items():
            metrics[f'benchmark_{key}'] = value
    
    # Filter metrics if custom_metrics is provided
    if custom_metrics is not None:
        metrics = {k: v for k, v in metrics.items() if k in custom_metrics}
    
    # Generate charts
    charts = {
        'equity_curve': create_equity_curve_chart(strategy_returns, benchmark_returns),
        'rolling_returns': create_rolling_returns_chart(strategy_returns),
        'monthly_heatmap': create_monthly_returns_heatmap(strategy_returns),
        'distribution': create_distribution_chart(strategy_returns)
    }
    
    return {
        'metrics': metrics,
        'charts': charts
    }

def display_backtest_metrics_dashboard(
    backtest_results: Dict[str, Any],
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark"
) -> None:
    """
    Display a complete backtest metrics dashboard in Streamlit
    
    Args:
        backtest_results: Results from generate_customizable_backtest_metrics
        strategy_name: Name of the strategy
        benchmark_name: Name of the benchmark (if applicable)
    """
    metrics = backtest_results['metrics']
    charts = backtest_results['charts']
    
    # Display key metrics in a nice grid
    st.markdown(f"### {strategy_name} Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        styled_metric_card("Total Return", metrics.get('total_return', 0), is_percent=True)
    
    with col2:
        styled_metric_card("CAGR", metrics.get('cagr', 0), is_percent=True)
    
    with col3:
        styled_metric_card("Sharpe Ratio", metrics.get('sharpe_ratio', 0))
    
    with col4:
        styled_metric_card("Max Drawdown", metrics.get('max_drawdown', 0), is_percent=True)
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        styled_metric_card("Win Rate", metrics.get('win_rate', 0), is_percent=True)
    
    with col2:
        styled_metric_card("Profit Factor", metrics.get('profit_factor', 0))
    
    with col3:
        styled_metric_card("Sortino Ratio", metrics.get('sortino_ratio', 0))
    
    with col4:
        styled_metric_card("Calmar Ratio", metrics.get('calmar_ratio', 0))
    
    # Display equity curve chart
    st.plotly_chart(charts['equity_curve'], use_container_width=True)
    
    # Create tabs for other charts
    tab1, tab2, tab3 = st.tabs(["Rolling Returns", "Monthly Returns", "Return Distribution"])
    
    with tab1:
        st.plotly_chart(charts['rolling_returns'], use_container_width=True)
    
    with tab2:
        st.plotly_chart(charts['monthly_heatmap'], use_container_width=True)
    
    with tab3:
        st.plotly_chart(charts['distribution'], use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Metrics")
    
    # Prepare metrics for display
    display_metrics = {}
    bench_metrics = {}
    
    for key, value in metrics.items():
        if key.startswith('benchmark_'):
            # Store benchmark metrics separately
            bench_key = key.replace('benchmark_', '')
            bench_metrics[bench_key] = value
        else:
            # Format based on metric type
            if key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor', 'recovery_factor', 'beta', 'information_ratio']:
                display_metrics[key] = f"{value:.2f}"
            elif key in ['total_return', 'cagr', 'volatility', 'max_drawdown', 'win_rate', 'avg_win', 'avg_loss', 'alpha']:
                display_metrics[key] = f"{value:.2f}%"
            elif key in ['max_win_streak', 'max_loss_streak']:
                display_metrics[key] = int(value)
            else:
                display_metrics[key] = value
    
    # Create comparison dataframe if benchmark metrics exist
    if bench_metrics:
        metrics_df = pd.DataFrame({
            'Metric': list(display_metrics.keys()),
            strategy_name: list(display_metrics.values()),
            benchmark_name: [bench_metrics.get(k.replace('benchmark_', ''), '-') for k in display_metrics.keys()]
        })
    else:
        metrics_df = pd.DataFrame({
            'Metric': list(display_metrics.keys()),
            'Value': list(display_metrics.values())
        })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

def get_mock_backtest_data():
    """Generate mock backtest data for demonstration"""
    # Create a date range for the past 2 years
    dates = pd.date_range(end=datetime.now(), periods=504, freq='B')  # 504 business days = ~2 years
    
    # Generate random daily returns with slight positive drift for strategy
    np.random.seed(42)  # For reproducibility
    strategy_returns = np.random.normal(0.0005, 0.012, len(dates))  # Slight positive drift, 1.2% daily volatility
    
    # Add some autocorrelation and momentum effects
    for i in range(5, len(strategy_returns)):
        strategy_returns[i] += 0.1 * np.mean(strategy_returns[i-5:i])
    
    # Generate random daily returns for benchmark (correlated with strategy but lower return)
    common_factor = np.random.normal(0.0002, 0.01, len(dates))
    specific_factor = np.random.normal(0.0, 0.008, len(dates))
    benchmark_returns = 0.7 * common_factor + 0.7 * specific_factor
    
    # Convert to pandas Series
    strategy_returns_series = pd.Series(strategy_returns, index=dates)
    benchmark_returns_series = pd.Series(benchmark_returns, index=dates)
    
    return strategy_returns_series, benchmark_returns_series
