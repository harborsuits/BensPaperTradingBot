"""
Visualization utilities for the BensBot Trading Dashboard
"""
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.theme import COLORS

def df_or_empty(data: any) -> pd.DataFrame:
    """Convert API data to a DataFrame or return empty DataFrame."""
    if isinstance(data, list) and data:
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()

def create_performance_chart(data: any) -> go.Figure:
    """Create a performance chart from portfolio data."""
    # This should be integrated with your existing portfolio history data
    # For now, we'll create a simple chart based on the data format you have
    
    # Assume we have daily returns for the last 30 days
    dates = pd.date_range(end=datetime.datetime.now(), periods=30)
    
    # Create some mock fluctuating values based on current portfolio value
    base_value = 10000  # Example base value
    if data and isinstance(data, list) and len(data) > 0:
        if 'total_value' in data[0]:
            base_value = data[0]['total_value']
        elif 'value' in data[0]:
            base_value = data[0]['value']
    
    # Generate slightly random performance data
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 30)
    cumulative_returns = np.cumprod(1 + daily_returns)
    values = base_value * cumulative_returns
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor=f"rgba({int(COLORS['primary'][1:3], 16)}, {int(COLORS['primary'][3:5], 16)}, {int(COLORS['primary'][5:7], 16)}, 0.1)"
    ))
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance (30 Days)',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='#ddd',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            showline=True,
            linecolor='#ddd',
            tickprefix='$',
            tickformat=',',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    return fig

def create_pie_chart(data: any) -> go.Figure:
    """Create a pie chart of asset allocation."""
    # This should integrate with your existing asset allocation data
    # For now, create an example based on your portfolio
    
    df = df_or_empty(data)
    
    # If we have actual data, try to use it
    if not df.empty and 'symbol' in df.columns and 'total_value' in df.columns:
        # Group by symbol and sum the values
        allocation_df = df.groupby('symbol')['total_value'].sum().reset_index()
        allocation_df.columns = ['Asset', 'Allocation']
    else:
        # Mock data if no real data available
        assets = {
            'Stocks': 45,
            'Crypto': 25,
            'Cash': 15,
            'Bonds': 10,
            'Other': 5
        }
        allocation_df = pd.DataFrame({
            'Asset': list(assets.keys()),
            'Allocation': list(assets.values())
        })
    
    # Create figure
    fig = go.Figure(data=[go.Pie(
        labels=allocation_df['Asset'],
        values=allocation_df['Allocation'],
        hole=.4,
        marker_colors=[COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning'], COLORS['info']]
    )])
    
    # Update layout
    fig.update_layout(
        title='Asset Allocation',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_trade_history_chart(trades: any) -> go.Figure:
    """Create a bar chart of recent trade performance."""
    # This should integrate with your existing trade history data
    
    df = df_or_empty(trades)
    
    if df.empty or ('profit_loss' not in df.columns and 'pnl' not in df.columns):
        # Generate mock data if no real data available
        dates = pd.date_range(end=datetime.datetime.now(), periods=10)
        profits = np.random.normal(100, 200, 10)
        symbols = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], 10)
        
        df = pd.DataFrame({
            'date': dates,
            'profit_loss': profits,
            'symbol': symbols
        })
    else:
        # Use at most the last 10 trades
        df = df.tail(10)
    
    # Determine profit column
    profit_col = 'profit_loss' if 'profit_loss' in df.columns else 'pnl' if 'pnl' in df.columns else None
    
    # Create figure
    fig = go.Figure()
    
    if profit_col:
        # Add trace
        fig.add_trace(go.Bar(
            # Convert to list so Plotly accepts it
            x=list(range(len(df))),
            y=df[profit_col],
            text=df['symbol'] if 'symbol' in df.columns else '',
            marker_color=[COLORS['success'] if p >= 0 else COLORS['danger'] for p in df[profit_col]],
        ))
    
        # Update layout
        fig.update_layout(
            title='Recent Trade Performance',
            xaxis_title='Trade #',
            yaxis_title='Profit/Loss ($)',
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='x unified',
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='#ddd',
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f0f0f0',
                showline=True,
                linecolor='#ddd',
                tickprefix='$',
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
    
    return fig

def create_event_system_chart(event_data: any) -> go.Figure:
    """Create a visualization of the event system activity"""
    # This should integrate with your existing event system data
    
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Mock data for now - this should be replaced with real event metrics
    timestamps = pd.date_range(end=datetime.datetime.now(), periods=60, freq='1min')
    event_counts = np.random.poisson(lam=5, size=60)
    queue_sizes = np.random.poisson(lam=2, size=60) + event_counts
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=timestamps, 
            y=event_counts,
            mode='lines',
            name='Events/min',
            line=dict(color=COLORS['primary'], width=2)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps, 
            y=queue_sizes,
            mode='lines',
            name='Queue Size',
            line=dict(color=COLORS['secondary'], width=2, dash='dot')
        ),
        secondary_y=True,
    )
    
    # Set axes titles
    fig.update_layout(
        title='Event System Activity',
        height=300,
        xaxis_title='Time',
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    fig.update_yaxes(title_text="Events/min", secondary_y=False)
    fig.update_yaxes(title_text="Queue Size", secondary_y=True)
    
    return fig

def enhance_dataframe(df: pd.DataFrame, column_config: dict = None) -> None:
    """Display an enhanced dataframe with better styling and formatting."""
    if df.empty:
        st.info("No data available")
        return
    
    # Apply default column configuration if none provided
    if column_config is None:
        column_config = {}
    
    # Common financial columns formatting
    if 'profit_loss' in df.columns and 'profit_loss' not in column_config:
        column_config['profit_loss'] = st.column_config.NumberColumn(
            "Profit/Loss",
            format="$%.2f",
            help="Profit or loss for this position"
        )
    
    if 'value' in df.columns and 'value' not in column_config:
        column_config['value'] = st.column_config.NumberColumn(
            "Value",
            format="$%.2f",
            help="Current value"
        )
    
    if 'pnl' in df.columns and 'pnl' not in column_config:
        column_config['pnl'] = st.column_config.NumberColumn(
            "P&L",
            format="$%.2f",
            help="Profit and loss"
        )
    
    # Apply time formatting
    for col in df.columns:
        if col.lower() in ['time', 'date', 'timestamp', 'created_at', 'updated_at', 'ts']:
            if col not in column_config:
                column_config[col] = st.column_config.DatetimeColumn(
                    col.capitalize(),
                    format="MMM DD, YYYY, hh:mm a",
                    help=f"Time of {col}"
                )
    
    # Display the enhanced dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
