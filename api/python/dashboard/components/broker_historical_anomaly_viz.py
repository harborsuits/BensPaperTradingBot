"""
Broker Historical Performance Panel - Anomaly Detection Visualization Component

This component provides anomaly detection visualization for historical broker performance data,
including z-score based outlier detection and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.dashboard.components.broker_historical_panel_base import format_metric_name


def detect_anomalies(
    df: pd.DataFrame,
    metric_name: str,
    window: int = 20,
    threshold: float = 2.5
) -> pd.DataFrame:
    """
    Detect anomalies in a time series using Z-score method
    
    Args:
        df: DataFrame with performance data
        metric_name: Name of metric to analyze
        window: Window size for rolling statistics
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with original data and anomaly flags
    """
    if df.empty or metric_name not in df.columns:
        return df
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Calculate rolling mean and std
    result[f'{metric_name}_mean'] = result[metric_name].rolling(window=window).mean()
    result[f'{metric_name}_std'] = result[metric_name].rolling(window=window).std()
    
    # Handle initial NaN values in rolling window
    result[f'{metric_name}_mean'] = result[f'{metric_name}_mean'].fillna(result[metric_name].mean())
    result[f'{metric_name}_std'] = result[f'{metric_name}_std'].fillna(result[metric_name].std())
    
    # Avoid division by zero
    result[f'{metric_name}_std'] = result[f'{metric_name}_std'].replace(0, result[metric_name].std() or 0.001)
    
    # Calculate z-scores
    result[f'{metric_name}_zscore'] = (result[metric_name] - result[f'{metric_name}_mean']) / result[f'{metric_name}_std']
    
    # Flag anomalies
    result[f'{metric_name}_anomaly'] = np.abs(result[f'{metric_name}_zscore']) > threshold
    
    return result


def render_anomaly_detection(
    data: Dict[str, pd.DataFrame],
    broker_names: Dict[str, str],
    metric_name: str
):
    """
    Render anomaly detection visualization for a metric
    
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
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Metric selector
        selected_metric = st.selectbox(
            "Select Metric",
            options=sorted(valid_metrics),
            index=sorted(valid_metrics).index(metric_name) if metric_name in valid_metrics else 0,
            format_func=format_metric_name,
            key="anomaly_metric"
        )
    
    with col2:
        # Window size
        window_size = st.slider(
            "Rolling Window Size",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Size of rolling window for calculating statistics"
        )
    
    with col3:
        # Threshold
        threshold = st.slider(
            "Anomaly Threshold (σ)",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.5,
            help="Z-score threshold for anomaly detection (standard deviations)"
        )
    
    # Process each broker's data
    anomaly_results = {}
    anomaly_counts = {}
    
    for broker_id, df in data.items():
        if selected_metric not in df.columns or len(df) < window_size:
            continue
        
        # Detect anomalies
        anomaly_df = detect_anomalies(
            df=df,
            metric_name=selected_metric,
            window=window_size,
            threshold=threshold
        )
        
        # Store results
        anomaly_results[broker_id] = anomaly_df
        
        # Count anomalies
        anomaly_counts[broker_id] = anomaly_df[f'{selected_metric}_anomaly'].sum()
    
    # Display anomaly visualization
    st.subheader("Anomaly Detection")
    
    # First, show summary of anomalies found
    if anomaly_counts:
        st.write("Anomalies detected per broker:")
        
        summary_data = []
        for broker_id, count in anomaly_counts.items():
            broker_name = broker_names.get(broker_id, broker_id)
            total_points = len(anomaly_results[broker_id])
            percentage = (count / total_points) * 100 if total_points > 0 else 0
            
            summary_data.append({
                "Broker": broker_name,
                "Anomalies": count,
                "Total Points": total_points,
                "Percentage": f"{percentage:.1f}%"
            })
        
        st.table(pd.DataFrame(summary_data))
    
    # Create visualization for each broker
    for broker_id, anomaly_df in anomaly_results.items():
        broker_name = broker_names.get(broker_id, broker_id)
        
        # Create figure
        fig = go.Figure()
        
        # Add original metric line
        fig.add_trace(go.Scatter(
            x=anomaly_df.index,
            y=anomaly_df[selected_metric],
            mode='lines',
            name=f"{broker_name} {format_metric_name(selected_metric)}",
            line=dict(color='blue', width=1.5)
        ))
        
        # Add rolling mean
        fig.add_trace(go.Scatter(
            x=anomaly_df.index,
            y=anomaly_df[f'{selected_metric}_mean'],
            mode='lines',
            name=f"Rolling Mean (n={window_size})",
            line=dict(color='green', width=1.5, dash='dash')
        ))
        
        # Add anomaly points
        anomaly_points = anomaly_df[anomaly_df[f'{selected_metric}_anomaly']]
        
        if not anomaly_points.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points[selected_metric],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x'
                )
            ))
        
        # Add confidence interval (mean ± threshold * std)
        fig.add_trace(go.Scatter(
            x=anomaly_df.index,
            y=anomaly_df[f'{selected_metric}_mean'] + threshold * anomaly_df[f'{selected_metric}_std'],
            mode='lines',
            name=f'Upper Bound ({threshold}σ)',
            line=dict(color='rgba(255,0,0,0.2)', width=1, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=anomaly_df.index,
            y=anomaly_df[f'{selected_metric}_mean'] - threshold * anomaly_df[f'{selected_metric}_std'],
            mode='lines',
            name=f'Lower Bound ({threshold}σ)',
            line=dict(color='rgba(255,0,0,0.2)', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(0,100,0,0.1)'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Anomaly Detection: {broker_name} - {format_metric_name(selected_metric)}",
            xaxis_title="Date/Time",
            yaxis_title=format_metric_name(selected_metric),
            height=400,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
    
    # Display anomaly table for the first broker (if any)
    if anomaly_results and any(anomaly_counts.values()):
        st.subheader("Anomaly Details")
        
        # Find broker with most anomalies
        broker_id = max(anomaly_counts, key=anomaly_counts.get)
        broker_name = broker_names.get(broker_id, broker_id)
        anomaly_df = anomaly_results[broker_id]
        
        # Get anomaly points
        anomaly_points = anomaly_df[anomaly_df[f'{selected_metric}_anomaly']]
        
        if not anomaly_points.empty:
            # Format for display
            anomaly_table = pd.DataFrame({
                'Timestamp': anomaly_points.index,
                'Value': anomaly_points[selected_metric],
                'Expected (Mean)': anomaly_points[f'{selected_metric}_mean'],
                'Z-Score': anomaly_points[f'{selected_metric}_zscore'],
            })
            
            # Sort by absolute z-score (most extreme first)
            anomaly_table = anomaly_table.sort_values(by='Z-Score', key=abs, ascending=False)
            
            # Format values
            anomaly_table['Value'] = anomaly_table['Value'].round(2)
            anomaly_table['Expected (Mean)'] = anomaly_table['Expected (Mean)'].round(2)
            anomaly_table['Z-Score'] = anomaly_table['Z-Score'].round(2)
            
            # Show only top anomalies if there are many
            if len(anomaly_table) > 10:
                st.write(f"Top 10 anomalies for {broker_name} (out of {len(anomaly_table)}):")
                st.dataframe(anomaly_table.head(10))
            else:
                st.write(f"All anomalies for {broker_name}:")
                st.dataframe(anomaly_table)
        else:
            st.info(f"No anomalies found for {broker_name} with current settings")
    
    # Show interpretation
    with st.expander("Interpreting Anomalies", expanded=False):
        st.markdown("""
        ### Understanding Anomaly Detection
        
        This visualization uses statistical methods to identify unusual patterns in broker performance:
        
        - **Rolling Mean**: The average value over the specified window
        - **Confidence Interval**: Range of expected values (mean ± threshold × standard deviation)
        - **Anomalies**: Points that exceed the confidence interval (outliers)
        
        ### What to Look For
        
        - **Isolated Anomalies**: Single points outside the confidence interval may indicate temporary issues
        - **Clusters of Anomalies**: Multiple consecutive anomalies suggest sustained problems
        - **Pattern Changes**: Sudden shifts in the pattern could indicate infrastructure changes
        
        ### Actions to Consider
        
        - Investigate individual anomalies to determine root causes
        - Correlate anomalies with known events (deployments, market volatility)
        - Monitor clusters of anomalies that may precede broker failures
        """)


def render_broker_anomaly_analysis(data_service=None):
    """
    Render anomaly detection for historical broker performance
    
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
    
    # Render anomaly detection visualization
    render_anomaly_detection(data, broker_names, "latency_mean_ms")


if __name__ == "__main__":
    # For local testing only
    render_broker_anomaly_analysis()
