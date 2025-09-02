"""
Broker Historical Performance Panel - Trend Visualization Component

This component provides trend visualization for the historical broker performance data,
including time series plots, moving averages, and trend analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.dashboard.components.broker_historical_panel_base import format_metric_name


def calculate_moving_averages(
    df: pd.DataFrame,
    metric_name: str,
    windows: List[int] = [5, 20]
) -> pd.DataFrame:
    """
    Calculate moving averages for a metric
    
    Args:
        df: DataFrame with performance data
        metric_name: Name of metric to analyze
        windows: List of window sizes for moving averages
        
    Returns:
        DataFrame with original metric and moving averages
    """
    if df.empty or metric_name not in df.columns:
        return df
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate moving averages
    for window in windows:
        result[f'{metric_name}_ma{window}'] = result[metric_name].rolling(window=window).mean()
    
    return result


def render_trend_time_series(
    data: Dict[str, pd.DataFrame],
    broker_names: Dict[str, str],
    metric_name: str
):
    """
    Render time series trend visualization for a metric
    
    Args:
        data: Dict mapping broker_id to DataFrame of performance data
        broker_names: Dict mapping broker_id to display name
        metric_name: Name of metric to visualize
    """
    # Get valid metric for available data
    valid_metrics = set()
    for broker_id, df in data.items():
        valid_metrics.update(col for col in df.columns if isinstance(df[col].dtype, (np.float64, np.int64)) or df[col].dtype in [float, int])
    
    if not valid_metrics:
        st.warning("No numeric metrics found in data")
        return
    
    # If provided metric not valid, use first valid metric
    if metric_name not in valid_metrics:
        metric_name = next(iter(valid_metrics))
    
    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric",
        options=sorted(valid_metrics),
        index=sorted(valid_metrics).index(metric_name) if metric_name in valid_metrics else 0,
        format_func=format_metric_name
    )
    
    # Moving average options
    show_ma = st.checkbox("Show Moving Averages", value=True)
    
    if show_ma:
        col1, col2 = st.columns(2)
        with col1:
            ma_windows = st.multiselect(
                "Moving Average Windows",
                options=[5, 10, 20, 50, 100],
                default=[5, 20]
            )
        with col2:
            show_trend_line = st.checkbox("Show Trend Line", value=True)
    else:
        ma_windows = []
        show_trend_line = False
    
    # Prepare figure
    fig = go.Figure()
    
    # Add time series for each broker
    for broker_id, df in data.items():
        if selected_metric not in df.columns:
            continue
        
        broker_name = broker_names.get(broker_id, broker_id)
        
        # Add raw data series
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[selected_metric],
            mode='lines',
            name=f"{broker_name}",
            line=dict(width=2)
        ))
        
        # Add moving averages if requested
        if show_ma and ma_windows:
            # Calculate moving averages
            df_ma = calculate_moving_averages(df, selected_metric, ma_windows)
            
            # Add to plot
            for window in ma_windows:
                ma_col = f'{selected_metric}_ma{window}'
                if ma_col in df_ma.columns:
                    fig.add_trace(go.Scatter(
                        x=df_ma.index,
                        y=df_ma[ma_col],
                        mode='lines',
                        name=f"{broker_name} (MA-{window})",
                        line=dict(width=1.5, dash='dash')
                    ))
            
            # Add trend line if requested
            if show_trend_line and len(df) > 5:
                # Simple linear regression for trend
                x = np.arange(len(df))
                y = df[selected_metric].values
                
                # Calculate trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Add trend line
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=p(x),
                    mode='lines',
                    name=f"{broker_name} (Trend)",
                    line=dict(width=1, dash='dot')
                ))
    
    # Update layout
    fig.update_layout(
        title=f"{format_metric_name(selected_metric)} Over Time",
        xaxis_title="Date/Time",
        yaxis_title=format_metric_name(selected_metric),
        legend_title="Broker",
        height=500,
        hovermode="x unified"
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display trend summary
    if len(data) > 0:
        st.subheader("Trend Summary")
        
        trend_data = []
        for broker_id, df in data.items():
            if selected_metric not in df.columns or len(df) < 2:
                continue
            
            broker_name = broker_names.get(broker_id, broker_id)
            
            # Calculate basic trend stats
            first_value = df[selected_metric].iloc[0]
            last_value = df[selected_metric].iloc[-1]
            change = last_value - first_value
            pct_change = (change / first_value) * 100 if first_value != 0 else float('inf')
            
            # Simple trend direction
            if pct_change > 5:
                trend = "🔺 Improving"
            elif pct_change < -5:
                trend = "🔻 Degrading"
            else:
                trend = "➡️ Stable"
            
            trend_data.append({
                "Broker": broker_name,
                "Start Value": f"{first_value:.2f}",
                "End Value": f"{last_value:.2f}",
                "Change": f"{change:.2f} ({pct_change:.1f}%)",
                "Trend": trend
            })
        
        if trend_data:
            st.table(pd.DataFrame(trend_data))


def render_seasonality_analysis(
    data: Dict[str, pd.DataFrame],
    broker_names: Dict[str, str],
    metric_name: str
):
    """
    Render seasonality analysis visualization for a metric
    
    Args:
        data: Dict mapping broker_id to DataFrame of performance data
        broker_names: Dict mapping broker_id to display name
        metric_name: Name of metric to visualize
    """
    # Get valid metric for available data
    valid_metrics = set()
    for broker_id, df in data.items():
        valid_metrics.update(col for col in df.columns if isinstance(df[col].dtype, (np.float64, np.int64)) or df[col].dtype in [float, int])
    
    if not valid_metrics:
        st.warning("No numeric metrics found in data")
        return
    
    # If provided metric not valid, use first valid metric
    if metric_name not in valid_metrics:
        metric_name = next(iter(valid_metrics))
    
    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric for Seasonality",
        options=sorted(valid_metrics),
        index=sorted(valid_metrics).index(metric_name) if metric_name in valid_metrics else 0,
        format_func=format_metric_name,
        key="seasonality_metric"
    )
    
    # Seasonality type selector
    seasonality_type = st.selectbox(
        "Seasonality Type",
        options=["Hourly", "Daily", "Weekly"],
        index=0
    )
    
    # Prepare figure
    fig = go.Figure()
    
    # Add series for each broker
    for broker_id, df in data.items():
        if selected_metric not in df.columns or len(df) < 24:  # Require at least 24 data points
            continue
        
        broker_name = broker_names.get(broker_id, broker_id)
        
        # Extract seasonality based on selection
        if seasonality_type == "Hourly":
            # Group by hour of day
            df['hour'] = df.index.hour
            seasonal = df.groupby('hour')[selected_metric].mean()
            
            fig.add_trace(go.Scatter(
                x=seasonal.index,
                y=seasonal.values,
                mode='lines+markers',
                name=broker_name
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Hourly Pattern: {format_metric_name(selected_metric)}",
                xaxis_title="Hour of Day",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(24)),
                    ticktext=[f"{h:02d}:00" for h in range(24)]
                )
            )
            
        elif seasonality_type == "Daily":
            # Group by day of week
            df['day'] = df.index.day_name()
            seasonal = df.groupby('day')[selected_metric].mean()
            
            # Reorder days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            seasonal = seasonal.reindex(days_order)
            
            fig.add_trace(go.Scatter(
                x=seasonal.index,
                y=seasonal.values,
                mode='lines+markers',
                name=broker_name
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Daily Pattern: {format_metric_name(selected_metric)}",
                xaxis_title="Day of Week"
            )
            
        elif seasonality_type == "Weekly":
            # Group by day of month
            df['day_of_month'] = df.index.day
            seasonal = df.groupby('day_of_month')[selected_metric].mean()
            
            fig.add_trace(go.Scatter(
                x=seasonal.index,
                y=seasonal.values,
                mode='lines+markers',
                name=broker_name
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Monthly Pattern: {format_metric_name(selected_metric)}",
                xaxis_title="Day of Month"
            )
    
    # Update common layout properties
    fig.update_layout(
        yaxis_title=format_metric_name(selected_metric),
        legend_title="Broker",
        height=400,
        hovermode="x unified"
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display interpretation
    if seasonality_type == "Hourly":
        st.info("""
        **Interpreting Hourly Patterns:** 
        
        This chart shows how metrics vary by hour of the day. Look for:
        - Peak performance times
        - Low performance periods
        - Regular patterns that could indicate market open/close effects
        - Differences between brokers in handling peak hours
        """)
    elif seasonality_type == "Daily":
        st.info("""
        **Interpreting Daily Patterns:** 
        
        This chart shows how metrics vary by day of the week. Look for:
        - Weekday vs weekend differences
        - Higher volatility days
        - Regular maintenance windows
        - Days when particular brokers consistently underperform
        """)
    elif seasonality_type == "Weekly":
        st.info("""
        **Interpreting Monthly Patterns:** 
        
        This chart shows how metrics vary by day of the month. Look for:
        - End-of-month effects
        - Settlement day impacts
        - Monthly maintenance patterns
        - Consistent monthly cycles
        """)


def render_broker_trend_analysis(data_service=None):
    """
    Render trend analysis for historical broker performance
    
    Args:
        data_service: Optional DataService instance (not used here, 
                     as we're using data from session state)
    """
    # Check if data is available in session state
    if not hasattr(st.session_state, 'broker_historical_data') or not st.session_state.broker_historical_data:
        st.warning("Please load broker historical data first")
        return
    
    # Get data from session state
    data = st.session_state.broker_historical_data
    broker_names = st.session_state.broker_historical_names
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Time Series Trends", "Seasonality Analysis"])
    
    # Tab 1: Time Series Trends
    with tabs[0]:
        render_trend_time_series(data, broker_names, "latency_mean_ms")
    
    # Tab 2: Seasonality Analysis
    with tabs[1]:
        render_seasonality_analysis(data, broker_names, "latency_mean_ms")


if __name__ == "__main__":
    # For local testing only
    render_broker_trend_analysis()
