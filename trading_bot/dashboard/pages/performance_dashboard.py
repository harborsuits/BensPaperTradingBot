"""
Performance Monitoring Dashboard

This dashboard provides trading performance analytics, P&L breakdowns,
and strategy comparisons.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from trading_bot.core.event_bus import get_global_event_bus


def render_daily_pnl_chart():
    """Render daily P&L chart with cumulative performance."""
    # Sample data - would be replaced with actual trading results
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    # Generate some realistic daily P&L values with a slight positive bias
    np.random.seed(42)  # For reproducible results
    daily_pnl = np.random.normal(200, 1000, len(dates))  # Mean $200, std $1000
    
    # Calculate cumulative P&L
    cumulative_pnl = np.cumsum(daily_pnl)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Daily P&L': daily_pnl,
        'Cumulative P&L': cumulative_pnl
    })
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Daily P&L", "Cumulative P&L"])
    
    with tab1:
        # Create bar chart for daily P&L
        fig = go.Figure()
        
        # Add green bars for positive values, red for negative
        positive_pnl = [max(0, pnl) for pnl in df['Daily P&L']]
        negative_pnl = [min(0, pnl) for pnl in df['Daily P&L']]
        
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=positive_pnl,
            name='Profit',
            marker_color='rgba(0, 153, 51, 0.7)'
        ))
        
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=negative_pnl,
            name='Loss',
            marker_color='rgba(204, 0, 0, 0.7)'
        ))
        
        fig.update_layout(
            title="Daily P&L",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            barmode='relative'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create line chart for cumulative P&L
        fig = px.line(
            df, 
            x='Date', 
            y='Cumulative P&L',
            title='Cumulative P&L Performance'
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=df['Date'].min(),
            y0=0,
            x1=df['Date'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Color the line based on whether it's above or below 0
        fig.update_traces(line=dict(color='green'))
        
        st.plotly_chart(fig, use_container_width=True)


def render_strategy_comparison():
    """Render strategy comparison metrics."""
    # Sample data - would be replaced with actual strategy metrics
    strategies = {
        'Mean Reversion': {'pnl': 2540.75, 'win_rate': 68.5, 'trades': 54, 'sharpe': 1.8},
        'Momentum': {'pnl': 1875.25, 'win_rate': 62.3, 'trades': 42, 'sharpe': 1.5},
        'Stat Arb': {'pnl': -540.50, 'win_rate': 48.2, 'trades': 28, 'sharpe': 0.7},
        'Trend Following': {'pnl': 3250.80, 'win_rate': 58.6, 'trades': 35, 'sharpe': 2.1}
    }
    
    # Create DataFrame
    df = pd.DataFrame({
        'Strategy': list(strategies.keys()),
        'P&L': [s['pnl'] for s in strategies.values()],
        'Win Rate (%)': [s['win_rate'] for s in strategies.values()],
        'Trades': [s['trades'] for s in strategies.values()],
        'Sharpe Ratio': [s['sharpe'] for s in strategies.values()]
    })
    
    # Add P&L per trade
    df['P&L per Trade'] = df['P&L'] / df['Trades']
    
    # Display metrics table
    st.write("### Strategy Performance Comparison")
    
    # Apply styling based on P&L
    def style_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
        return ''
    
    # Display the styled DataFrame
    styled_df = df.style.format({
        'P&L': '${:.2f}', 
        'Win Rate (%)': '{:.1f}%', 
        'P&L per Trade': '${:.2f}',
        'Sharpe Ratio': '{:.2f}'
    }).applymap(style_pnl, subset=['P&L', 'P&L per Trade'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Create bar chart comparing strategies
    fig = go.Figure()
    
    for metric in ['P&L', 'Win Rate (%)', 'Sharpe Ratio']:
        fig.add_trace(go.Bar(
            x=df['Strategy'],
            y=df[metric],
            name=metric
        ))
    
    fig.update_layout(
        title="Strategy Metrics Comparison",
        xaxis_title="Strategy",
        yaxis_title="Value",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_trade_analytics():
    """Render trade analytics section."""
    # Sample trade data
    trades = [
        {"Symbol": "AAPL", "Trades": 28, "Win Rate": 71.4, "Profit": 840.25, "Loss": -320.50, "Net": 519.75},
        {"Symbol": "MSFT", "Trades": 32, "Win Rate": 65.6, "Profit": 925.75, "Loss": -385.25, "Net": 540.50},
        {"Symbol": "GOOGL", "Trades": 18, "Win Rate": 55.6, "Profit": 685.50, "Loss": -510.25, "Net": 175.25},
        {"Symbol": "AMZN", "Trades": 22, "Win Rate": 63.6, "Profit": 742.25, "Loss": -365.50, "Net": 376.75},
        {"Symbol": "TSLA", "Trades": 24, "Win Rate": 58.3, "Profit": 1245.50, "Loss": -865.75, "Net": 379.75},
        {"Symbol": "NVDA", "Trades": 16, "Win Rate": 75.0, "Profit": 1125.50, "Loss": -225.25, "Net": 900.25},
        {"Symbol": "META", "Trades": 20, "Win Rate": 60.0, "Profit": 685.25, "Loss": -420.50, "Net": 264.75},
        {"Symbol": "AMD", "Trades": 14, "Win Rate": 64.3, "Profit": 542.75, "Loss": -285.25, "Net": 257.50}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(trades)
    
    # Calculate profit factor
    df['Profit Factor'] = df['Profit'].abs() / df['Loss'].abs()
    
    # Format columns
    formatted_df = df.copy()
    formatted_df['Win Rate'] = formatted_df['Win Rate'].map('{:.1f}%'.format)
    formatted_df['Profit'] = formatted_df['Profit'].map('${:.2f}'.format)
    formatted_df['Loss'] = formatted_df['Loss'].map('${:.2f}'.format)
    formatted_df['Net'] = formatted_df['Net'].map('${:.2f}'.format)
    formatted_df['Profit Factor'] = formatted_df['Profit Factor'].map('{:.2f}'.format)
    
    # Display the table
    st.write("### Symbol Performance")
    st.dataframe(formatted_df, use_container_width=True)
    
    # Create a bubble chart of symbol performance
    fig = px.scatter(
        df,
        x="Win Rate",
        y="Net",
        size="Trades",
        color="Profit Factor",
        hover_name="Symbol",
        size_max=60,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        title="Symbol Performance Analysis",
        xaxis_title="Win Rate (%)",
        yaxis_title="Net P&L ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_time_analysis():
    """Render time-based analysis of trading performance."""
    # Generate sample data for time-based analysis
    hours = ["9:30", "10:00", "10:30", "11:00", "11:30", "12:00", 
             "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
    
    # Average P&L by hour - higher in the morning and closing hour
    avg_pnl = [85.5, 65.2, 45.8, 25.3, 15.5, -10.2, -5.5, 12.3, 18.5, 22.7, 35.2, 55.8, 75.2, 95.5]
    
    # Create DataFrame
    hour_df = pd.DataFrame({
        'Hour': hours,
        'Average P&L': avg_pnl
    })
    
    # Create bar chart
    fig = go.Figure()
    
    # Color bars based on positive/negative values
    colors = ['green' if x >= 0 else 'red' for x in avg_pnl]
    
    fig.add_trace(go.Bar(
        x=hour_df['Hour'],
        y=hour_df['Average P&L'],
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Average P&L by Hour of Day",
        xaxis_title="Hour (ET)",
        yaxis_title="Average P&L ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add day of week analysis
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_pnl = [245.5, -125.8, 385.2, 175.5, 420.3]
    
    day_df = pd.DataFrame({
        'Day': days,
        'Average P&L': day_pnl
    })
    
    # Create bar chart
    fig = go.Figure()
    
    # Color bars based on positive/negative values
    colors = ['green' if x >= 0 else 'red' for x in day_pnl]
    
    fig.add_trace(go.Bar(
        x=day_df['Day'],
        y=day_df['Average P&L'],
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Average P&L by Day of Week",
        xaxis_title="Day",
        yaxis_title="Average P&L ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_key_metrics():
    """Render key performance metrics."""
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total P&L", "$3,245.75", "+2.35%")
    
    with col2:
        st.metric("Win Rate", "64.8%", "+4.2%")
    
    with col3:
        st.metric("Profit Factor", "1.85", "+0.15")
    
    with col4:
        st.metric("Sharpe Ratio", "1.62", "+0.08")


def main():
    """Render the performance dashboard."""
    st.set_page_config(
        page_title="Trading Performance",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Trading Performance Dashboard")
    st.write("Analysis and visualization of trading performance and strategy metrics")
    
    # Display date range information
    st.sidebar.write("**Data Range:** Last 30 days")
    st.sidebar.write(f"**As of:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.experimental_rerun()
    
    # Render key metrics
    render_key_metrics()
    
    # Add separator
    st.markdown("---")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "P&L Performance", 
        "Strategy Analysis",
        "Symbol Analysis",
        "Time Analysis"
    ])
    
    with tab1:
        render_daily_pnl_chart()
    
    with tab2:
        render_strategy_comparison()
    
    with tab3:
        render_trade_analytics()
    
    with tab4:
        render_time_analysis()
    
    # Footer disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This dashboard presents simulated trading data for demonstration purposes. "
        "Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
