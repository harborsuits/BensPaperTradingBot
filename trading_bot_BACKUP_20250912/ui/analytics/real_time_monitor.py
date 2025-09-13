import pandas as pd
import numpy as np
import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import threading

from ...strategies.testing.enhanced_component_tester import EnhancedComponentTestCase


class RealTimeMonitor:
    """
    Real-time monitoring for trading components with live performance tracking,
    signal visualization, and metric updates
    """
    
    @staticmethod
    def update_real_time_metrics(components, data, max_history=100):
        """
        Update real-time metrics for components
        
        Args:
            components: List of components to monitor
            data: Current market data
            max_history: Maximum history length
            
        Returns:
            Dict with updated metrics
        """
        # This would normally connect to live market data
        # For demo, we'll simulate using the latest data point
        
        current_time = datetime.datetime.now()
        symbol = data.index.name or "unknown"
        
        # Get latest data point
        latest_data = data.iloc[-1:].copy()
        
        # Set index to current time for latest point
        latest_data.index = [current_time]
        
        # Update metrics for each component
        updates = {}
        
        for component in components:
            component_id = component.component_id
            
            # Skip if component is not in session state
            if component_id not in st.session_state.real_time_metrics:
                continue
            
            # Get component metrics
            metrics = st.session_state.real_time_metrics[component_id]
            
            # Add timestamp
            if len(metrics['timestamps']) >= max_history:
                metrics['timestamps'] = metrics['timestamps'][1:] + [current_time]
            else:
                metrics['timestamps'].append(current_time)
            
            # Create test case for signal
            test_case = EnhancedComponentTestCase(component)
            test_case.with_historical_data(symbol, latest_data)
            
            # Run test
            test_case.run()
            
            # Get signal
            signal = 0.0
            try:
                # This will vary based on component type
                signal = test_case.get_signals(symbol).iloc[-1]
            except:
                signal = 0.0
            
            # Add signal
            if len(metrics['signals']) >= max_history:
                metrics['signals'] = metrics['signals'][1:] + [signal]
            else:
                metrics['signals'].append(signal)
            
            # Update performance metrics
            
            # We need a window of data for meaningful metrics 
            # Get trailing data window
            window_size = min(30, len(data))
            trailing_data = data.iloc[-window_size:].copy()
            
            # Create test case for metrics
            metric_test_case = EnhancedComponentTestCase(component)
            metric_test_case.with_historical_data(symbol, trailing_data)
            
            # Run test with metrics
            metric_test_case.run_with_metrics()
            
            # Get performance metrics
            performance = metric_test_case.performance_metrics.get(symbol, {})
            
            # Store performance
            metrics['performance'] = performance
            
            # Add to updates
            updates[component_id] = {
                'signal': signal,
                'performance': performance
            }
        
        return updates
    
    @staticmethod
    def display_real_time_metrics(selected_components):
        """
        Display real-time metrics for components
        
        Args:
            selected_components: List of component IDs to display
        """
        if not selected_components:
            st.info("No components selected for monitoring")
            return
        
        # Create tabs
        signal_tab, metrics_tab = st.tabs(["Real-time Signals", "Performance Metrics"])
        
        with signal_tab:
            RealTimeMonitor._display_real_time_signals(selected_components)
        
        with metrics_tab:
            RealTimeMonitor._display_real_time_performance(selected_components)
    
    @staticmethod
    def _display_real_time_signals(component_ids):
        """
        Display real-time signals for components
        
        Args:
            component_ids: List of component IDs to display
        """
        # Create signal chart
        fig = go.Figure()
        
        monitor_start_time = st.session_state.get('monitor_start_time', datetime.datetime.now())
        
        for component_id in component_ids:
            if component_id not in st.session_state.real_time_metrics:
                continue
            
            metrics = st.session_state.real_time_metrics[component_id]
            
            if not metrics['timestamps'] or not metrics['signals']:
                continue
            
            # Add signal trace
            fig.add_trace(go.Scatter(
                x=metrics['timestamps'],
                y=metrics['signals'],
                mode='lines',
                name=component_id
            ))
        
        # Update layout
        fig.update_layout(
            title="Real-time Component Signals",
            xaxis_title="Time",
            yaxis_title="Signal",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="minute", stepmode="backward"),
                        dict(count=5, label="5m", step="minute", stepmode="backward"),
                        dict(count=15, label="15m", step="minute", stepmode="backward"),
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current signal values
        st.subheader("Current Signal Values")
        
        # Create signal table
        rows = []
        
        for component_id in component_ids:
            if component_id not in st.session_state.real_time_metrics:
                continue
            
            metrics = st.session_state.real_time_metrics[component_id]
            
            if not metrics['signals']:
                continue
            
            rows.append({
                "Component": component_id,
                "Signal": metrics['signals'][-1],
                "Signal Range": f"{min(metrics['signals']):.4f} to {max(metrics['signals']):.4f}",
                "Signal Change": metrics['signals'][-1] - metrics['signals'][0] if len(metrics['signals']) > 1 else 0.0
            })
        
        if rows:
            signal_df = pd.DataFrame(rows)
            
            # Add styler to highlight positive/negative signals
            def highlight_signal(val):
                if isinstance(val, (int, float)):
                    color = "green" if val > 0 else "red" if val < 0 else "white"
                    return f"color: {color}"
                return ""
            
            # Display styled DataFrame
            st.dataframe(
                signal_df.style.applymap(highlight_signal, subset=["Signal", "Signal Change"]),
                use_container_width=True
            )
    
    @staticmethod
    def _display_real_time_performance(component_ids):
        """
        Display real-time performance metrics for components
        
        Args:
            component_ids: List of component IDs to display
        """
        # Select metrics to display
        metrics = [
            "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "total_returns"
        ]
        
        # Create metrics table
        rows = []
        
        for component_id in component_ids:
            if component_id not in st.session_state.real_time_metrics:
                continue
            
            component_metrics = st.session_state.real_time_metrics[component_id]
            
            if 'performance' not in component_metrics or not component_metrics['performance']:
                continue
            
            row = {"Component": component_id}
            
            for metric in metrics:
                if metric in component_metrics['performance']:
                    row[metric.replace('_', ' ').title()] = component_metrics['performance'][metric]
                else:
                    row[metric.replace('_', ' ').title()] = None
            
            rows.append(row)
        
        if rows:
            st.subheader("Current Performance Metrics")
            
            metrics_df = pd.DataFrame(rows)
            
            # Add styler to highlight positive/negative metrics
            def highlight_metric(val):
                if not isinstance(val, (int, float)) or pd.isna(val):
                    return ""
                
                # Specific styling for each metric
                if st.column_config.column_name == "Win Rate" or st.column_config.column_name == "Profit Factor" or \
                   st.column_config.column_name == "Sharpe Ratio" or st.column_config.column_name == "Total Returns":
                    color = "green" if val > 0 else "red" if val < 0 else "white"
                    return f"color: {color}"
                elif st.column_config.column_name == "Max Drawdown":
                    color = "red" if val < -0.1 else "orange" if val < -0.05 else "green"
                    return f"color: {color}"
                
                return ""
            
            # Display styled DataFrame
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create radar chart
            if len(metrics_df) > 0:
                RealTimeMonitor._plot_performance_radar(metrics_df, metrics)
    
    @staticmethod
    def _plot_performance_radar(df, metrics):
        """
        Plot radar chart for performance metrics
        
        Args:
            df: DataFrame with metrics
            metrics: List of metrics
        """
        if df.empty:
            return
        
        # Create normalized metrics for radar chart
        normalized_df = df.copy()
        
        for metric in metrics:
            metric_title = metric.replace('_', ' ').title()
            
            if metric_title not in normalized_df.columns:
                continue
            
            # Special handling for normalization
            if metric == "max_drawdown":
                # Lower is better for drawdown, so invert
                max_val = normalized_df[metric_title].max()
                min_val = normalized_df[metric_title].min()
                
                if max_val != min_val and not pd.isna(max_val) and not pd.isna(min_val):
                    normalized_df[metric_title] = 1.0 - ((normalized_df[metric_title] - min_val) / (max_val - min_val))
                else:
                    normalized_df[metric_title] = 1.0
            else:
                # Higher is better for most metrics
                max_val = normalized_df[metric_title].max()
                min_val = normalized_df[metric_title].min()
                
                if max_val != min_val and not pd.isna(max_val) and not pd.isna(min_val):
                    normalized_df[metric_title] = (normalized_df[metric_title] - min_val) / (max_val - min_val)
                else:
                    normalized_df[metric_title] = 1.0
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in normalized_df.iterrows():
            component = row['Component']
            
            # Add radar trace
            metric_values = []
            metric_names = []
            
            for metric in metrics:
                metric_title = metric.replace('_', ' ').title()
                
                if metric_title in row and not pd.isna(row[metric_title]):
                    metric_values.append(row[metric_title])
                    metric_names.append(metric_title)
            
            if metric_values:
                fig.add_trace(go.Scatterpolar(
                    r=metric_values,
                    theta=metric_names,
                    fill='toself',
                    name=component
                ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Component Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)


class MonitoringThread(threading.Thread):
    """Thread for background monitoring of component performance"""
    
    def __init__(self, components, data_source, update_interval=5, max_history=100):
        """
        Initialize monitoring thread
        
        Args:
            components: List of components to monitor
            data_source: Function to retrieve latest data
            update_interval: Update interval in seconds
            max_history: Maximum history length
        """
        super().__init__()
        self.daemon = True
        self.components = components
        self.data_source = data_source
        self.update_interval = update_interval
        self.max_history = max_history
        self.running = True
    
    def run(self):
        """Run the monitoring thread"""
        while self.running:
            try:
                # Get latest data
                data = self.data_source()
                
                if data is not None and not data.empty:
                    # Update metrics
                    RealTimeMonitor.update_real_time_metrics(
                        self.components,
                        data,
                        self.max_history
                    )
                
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
            
            # Sleep for update interval
            time.sleep(self.update_interval)
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
