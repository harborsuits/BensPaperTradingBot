import os
import json
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union

from ..component_registry import ComponentRegistry, ComponentType
from ..strategies.testing.enhanced_component_tester import EnhancedComponentTestCase
from .analytics.performance_tracker import PerformanceTracker
from .analytics.correlation_analyzer import CorrelationAnalyzer
from .analytics.real_time_monitor import RealTimeMonitor, MonitoringThread


class AnalyticsDashboardIntegration:
    """
    Analytics dashboard integration for trading components with performance tracking,
    component comparison, correlation analysis, and real-time monitoring.
    """
    
    def __init__(self, registry: ComponentRegistry):
        """
        Initialize the analytics dashboard integration
        
        Args:
            registry: Component registry
        """
        self.registry = registry
        self.data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.results_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
        
        # Create component instances
        self.performance_tracker = PerformanceTracker(self.data_folder, self.results_folder)
        
        # Ensure session state
        if 'component_performance' not in st.session_state:
            st.session_state.component_performance = {}
        
        if 'real_time_metrics' not in st.session_state:
            st.session_state.real_time_metrics = {}
            
        if 'component_history' not in st.session_state:
            st.session_state.component_history = {}
            
        if 'monitoring_threads' not in st.session_state:
            st.session_state.monitoring_threads = {}
    
    def render(self):
        """Render the analytics dashboard integration"""
        st.title("Trading Component Analytics")
        
        # Create tabs
        tabs = st.tabs([
            "Performance Tracking", 
            "Component Comparison", 
            "Correlation Analysis",
            "Real-time Monitoring"
        ])
        
        # Performance Tracking tab
        with tabs[0]:
            self._render_performance_tracking()
        
        # Component Comparison tab
        with tabs[1]:
            self._render_component_comparison()
        
        # Correlation Analysis tab
        with tabs[2]:
            self._render_correlation_analysis()
        
        # Real-time Monitoring tab
        with tabs[3]:
            self._render_real_time_monitoring()
    
    def _render_performance_tracking(self):
        """Render the performance tracking tab"""
        st.header("Component Performance Tracking")
        
        # Component selection
        component_type = st.selectbox(
            "Component Type",
            options=[ct.name for ct in ComponentType],
            key="perf_component_type"
        )
        
        # Get components of selected type
        comp_type = ComponentType[component_type]
        components = self.registry.get_components_by_type(comp_type)
        
        if not components:
            st.warning(f"No {component_type.lower().replace('_', ' ')} components found")
            return
        
        component_id = st.selectbox(
            "Component",
            options=[c.component_id for c in components],
            key="perf_component_id"
        )
        
        # Get selected component
        component = next((c for c in components if c.component_id == component_id), None)
        
        if not component:
            st.warning(f"Component {component_id} not found")
            return
        
        # Display component info
        st.subheader(f"Performance for {component_id}")
        
        # Parameters section
        with st.expander("Component Parameters", expanded=True):
            params = self.performance_tracker.get_component_parameters(component)
            
            # Create parameter table
            param_df = pd.DataFrame([
                {"Parameter": param, "Value": value}
                for param, value in params.items()
            ])
            
            st.dataframe(param_df, use_container_width=True)
        
        # Historical performance
        self._render_component_performance_history(component)
        
        # Run new performance test
        with st.expander("Run Performance Test", expanded=True):
            self._render_performance_test_form(component)
    
    def _render_component_performance_history(self, component):
        """Render component performance history"""
        st.subheader("Performance History")
        
        # Check if we have performance history for this component
        component_id = component.component_id
        component_history = st.session_state.component_history.get(component_id, [])
        
        if not component_history:
            # Load performance history from disk
            history = self.performance_tracker.load_component_performance_history(component_id)
            
            if not history:
                st.info("No performance history available for this component")
                return
                
            component_history = history
            st.session_state.component_history[component_id] = component_history
        
        # Display performance metrics over time
        metrics_over_time = pd.DataFrame(component_history)
        
        if not metrics_over_time.empty:
            # Time series of key metrics
            st.subheader("Performance Metrics Over Time")
            
            # Select metrics to display
            available_metrics = [col for col in metrics_over_time.columns if col not in ['timestamp', 'component_id', 'parameters']]
            
            if available_metrics:
                selected_metrics = st.multiselect(
                    "Select metrics to display",
                    options=available_metrics,
                    default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics,
                    key="history_metrics"
                )
                
                if selected_metrics:
                    # Convert timestamp to datetime
                    metrics_over_time['timestamp'] = pd.to_datetime(metrics_over_time['timestamp'])
                    
                    # Create time series plot
                    fig = px.line(
                        metrics_over_time, 
                        x='timestamp', 
                        y=selected_metrics,
                        title="Component Performance Metrics Over Time"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics statistics
                    st.subheader("Performance Statistics")
                    metrics_stats = metrics_over_time[selected_metrics].describe()
                    st.dataframe(metrics_stats, use_container_width=True)
                    
                    # Parameter changes over time
                    if 'parameters' in metrics_over_time.columns:
                        st.subheader("Parameter Changes")
                        
                        # Extract parameters as separate columns
                        param_history = pd.DataFrame(metrics_over_time['parameters'].tolist())
                        param_history['timestamp'] = metrics_over_time['timestamp']
                        
                        # Select parameters to display
                        available_params = [col for col in param_history.columns if col != 'timestamp']
                        
                        if available_params:
                            selected_params = st.multiselect(
                                "Select parameters to display",
                                options=available_params,
                                default=available_params[:3] if len(available_params) > 3 else available_params,
                                key="history_params"
                            )
                            
                            if selected_params:
                                # Create parameter history plot
                                fig = px.line(
                                    param_history, 
                                    x='timestamp', 
                                    y=selected_params,
                                    title="Parameter Changes Over Time"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No metrics data available in the history")
        else:
            st.info("No performance history available for this component")
    
    def _render_performance_test_form(self, component):
        """Render form for performance testing"""
        # Form for performance test
        with st.form(key="performance_test_form"):
            # Data selection
            data_files = self.performance_tracker.find_available_data()
            
            if not data_files:
                st.warning("No historical data files found")
                st.form_submit_button("Run Test", disabled=True)
                return
            
            # Display absolute paths but store relative for display clarity
            data_options = [os.path.basename(f) for f in data_files]
            data_indices = st.multiselect(
                "Select historical data files",
                options=range(len(data_options)),
                format_func=lambda i: data_options[i],
                key="perf_data_files"
            )
            
            selected_files = [data_files[i] for i in data_indices]
            
            # Test parameters
            test_name = st.text_input("Test Name", value=f"{component.component_id}_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Advanced options
            with st.expander("Advanced Options"):
                edge_cases = st.checkbox("Include edge case tests", value=True)
                error_cases = st.checkbox("Include error/boundary condition tests", value=True)
                
                test_periods = st.radio(
                    "Test Periods",
                    options=["All", "Last 30 Days", "Last 90 Days", "Last Year", "Custom"],
                    index=0,
                    horizontal=True
                )
                
                if test_periods == "Custom":
                    date_range = st.date_input(
                        "Select date range",
                        value=(datetime.date.today() - datetime.timedelta(days=90), datetime.date.today()),
                        key="perf_date_range"
                    )
            
            # Submit button
            submitted = st.form_submit_button("Run Performance Test")
            
            if submitted and selected_files:
                # Run performance test
                with st.spinner("Running performance test..."):
                    results = self.performance_tracker.run_performance_test(
                        component,
                        selected_files,
                        test_name,
                        edge_cases,
                        error_cases,
                        test_periods,
                        date_range if test_periods == "Custom" else None
                    )
                    
                    if results:
                        st.success("Performance test completed successfully")
                        
                        # Display results
                        self.performance_tracker.display_performance_test_results(results)
                    else:
                        st.error("Performance test failed")
    
    def _render_component_comparison(self):
        """Render the component comparison tab"""
        st.header("Component Comparison")
        
        # Component type selection
        component_type = st.selectbox(
            "Component Type",
            options=[ct.name for ct in ComponentType],
            key="comp_component_type"
        )
        
        # Get components of selected type
        comp_type = ComponentType[component_type]
        components = self.registry.get_components_by_type(comp_type)
        
        if not components:
            st.warning(f"No {component_type.lower().replace('_', ' ')} components found")
            return
        
        # Select components to compare
        component_ids = [c.component_id for c in components]
        selected_components = st.multiselect(
            "Select components to compare",
            options=component_ids,
            key="comp_component_ids"
        )
        
        if not selected_components:
            st.info("Select components to compare")
            return
        
        # Get selected components
        selected_comp_objects = [next((c for c in components if c.component_id == comp_id), None) for comp_id in selected_components]
        selected_comp_objects = [c for c in selected_comp_objects if c]  # Remove None
        
        if not selected_comp_objects:
            st.warning("No valid components selected")
            return
        
        # Data selection for comparison
        data_files = self.performance_tracker.find_available_data()
        
        if not data_files:
            st.warning("No historical data files found")
            return
        
        # Display absolute paths but store relative for display clarity
        data_options = [os.path.basename(f) for f in data_files]
        data_indices = st.multiselect(
            "Select historical data for comparison",
            options=range(len(data_options)),
            format_func=lambda i: data_options[i],
            key="comp_data_files"
        )
        
        if not data_indices:
            st.info("Select data files for comparison")
            return
        
        selected_files = [data_files[i] for i in data_indices]
        
        # Metrics selection
        metrics = [
            "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", 
            "total_returns", "volatility", "sortino_ratio", "calmar_ratio"
        ]
        
        selected_metrics = st.multiselect(
            "Select metrics for comparison",
            options=metrics,
            default=["win_rate", "profit_factor", "sharpe_ratio"],
            key="comp_metrics"
        )
        
        if not selected_metrics:
            st.info("Select metrics for comparison")
            return
        
        # Run comparison button
        if st.button("Run Comparison"):
            with st.spinner("Running component comparison..."):
                comparison_results = self.performance_tracker.run_component_comparison(
                    selected_comp_objects,
                    selected_files,
                    selected_metrics
                )
                
                if comparison_results:
                    self._display_comparison_results(comparison_results, selected_metrics)
                else:
                    st.error("Component comparison failed")
    
    def _display_comparison_results(self, results, metrics):
        """
        Display component comparison results
        
        Args:
            results: Dict with comparison results
            metrics: List of metrics compared
        """
        if not results or not results['data']:
            st.warning("No comparison results available")
            return
        
        # Create tabs for each symbol
        symbols = results['symbols']
        components = results['components']
        
        symbol_tabs = st.tabs(symbols)
        
        for i, symbol in enumerate(symbols):
            with symbol_tabs[i]:
                if symbol not in results['data']:
                    st.info(f"No data available for {symbol}")
                    continue
                
                symbol_data = results['data'][symbol]
                
                # Create side-by-side tables and charts for each metric
                for metric in metrics:
                    st.subheader(f"{metric.replace('_', ' ').title()}")
                    
                    # Extract metric values for each component
                    metric_values = {}
                    for component_id in components:
                        if component_id in symbol_data:
                            metric_values[component_id] = symbol_data[component_id].get(metric, 0.0)
                    
                    # Create bar chart
                    fig = px.bar(
                        x=list(metric_values.keys()),
                        y=list(metric_values.values()),
                        labels={'x': 'Component', 'y': metric.replace('_', ' ').title()},
                        title=f"{metric.replace('_', ' ').title()} Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create metric table
                    metric_df = pd.DataFrame([
                        {"Component": comp_id, metric.replace('_', ' ').title(): value}
                        for comp_id, value in metric_values.items()
                    ])
                    
                    st.dataframe(metric_df, use_container_width=True)
                
                # Create radar chart comparing all components
                self._plot_comparison_radar(symbol_data, components, metrics, symbol)
    
    def _plot_comparison_radar(self, symbol_data, components, metrics, title):
        """
        Plot radar chart comparing components
        
        Args:
            symbol_data: Dict with component metrics
            components: List of component IDs
            metrics: List of metrics
            title: Chart title
        """
        if not symbol_data or not components or not metrics:
            return
        
        # Create radar chart
        fig = go.Figure()
        
        # Normalize metrics for radar chart
        normalized_data = {}
        
        # Find min/max for each metric
        metric_min = {metric: float('inf') for metric in metrics}
        metric_max = {metric: float('-inf') for metric in metrics}
        
        for component_id in components:
            if component_id not in symbol_data:
                continue
                
            component_metrics = symbol_data[component_id]
            
            for metric in metrics:
                if metric in component_metrics:
                    value = component_metrics[metric]
                    metric_min[metric] = min(metric_min[metric], value)
                    metric_max[metric] = max(metric_max[metric], value)
        
        # Normalize each component's metrics
        for component_id in components:
            if component_id not in symbol_data:
                continue
                
            component_metrics = symbol_data[component_id]
            normalized_data[component_id] = {}
            
            for metric in metrics:
                if metric in component_metrics:
                    value = component_metrics[metric]
                    
                    # Handle special cases for normalization
                    if metric == "max_drawdown":
                        # Lower is better for drawdown
                        if metric_min[metric] == metric_max[metric]:
                            normalized_data[component_id][metric] = 1.0
                        else:
                            normalized_data[component_id][metric] = 1.0 - ((value - metric_min[metric]) / (metric_max[metric] - metric_min[metric]))
                    else:
                        # Higher is better for most metrics
                        if metric_min[metric] == metric_max[metric]:
                            normalized_data[component_id][metric] = 1.0
                        else:
                            normalized_data[component_id][metric] = (value - metric_min[metric]) / (metric_max[metric] - metric_min[metric])
        
        # Add trace for each component
        for component_id in components:
            if component_id not in normalized_data:
                continue
                
            component_metrics = normalized_data[component_id]
            
            # Add radar trace
            fig.add_trace(go.Scatterpolar(
                r=[component_metrics.get(metric, 0.0) for metric in metrics],
                theta=metrics,
                fill='toself',
                name=component_id
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Component Comparison: {title}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add legend with original values
        st.subheader("Original Metric Values")
        
        # Create table with original values
        rows = []
        
        for component_id in components:
            if component_id not in symbol_data:
                continue
                
            row = {"Component": component_id}
            
            for metric in metrics:
                if metric in symbol_data[component_id]:
                    row[metric.replace('_', ' ').title()] = symbol_data[component_id][metric]
                else:
                    row[metric.replace('_', ' ').title()] = 0.0
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        st.dataframe(df, use_container_width=True)
    
    def _render_correlation_analysis(self):
        """Render the correlation analysis tab"""
        st.header("Correlation Analysis")
        
        # Component type selection
        component_type = st.selectbox(
            "Component Type",
            options=[ct.name for ct in ComponentType],
            key="corr_component_type"
        )
        
        # Get components of selected type
        comp_type = ComponentType[component_type]
        components = self.registry.get_components_by_type(comp_type)
        
        if not components:
            st.warning(f"No {component_type.lower().replace('_', ' ')} components found")
            return
        
        # Select components for correlation analysis
        component_ids = [c.component_id for c in components]
        selected_components = st.multiselect(
            "Select components for correlation analysis",
            options=component_ids,
            key="corr_component_ids"
        )
        
        if not selected_components:
            st.info("Select components for correlation analysis")
            return
        
        # Get selected components
        selected_comp_objects = [next((c for c in components if c.component_id == comp_id), None) for comp_id in selected_components]
        selected_comp_objects = [c for c in selected_comp_objects if c]  # Remove None
        
        if not selected_comp_objects:
            st.warning("No valid components selected")
            return
        
        # Data selection for correlation
        data_files = self.performance_tracker.find_available_data()
        
        if not data_files:
            st.warning("No historical data files found")
            return
        
        # Display absolute paths but store relative for display clarity
        data_options = [os.path.basename(f) for f in data_files]
        data_index = st.selectbox(
            "Select historical data for correlation analysis",
            options=range(len(data_options)),
            format_func=lambda i: data_options[i],
            key="corr_data_file"
        )
        
        selected_file = data_files[data_index]
        
        # Correlation type selection
        correlation_type = st.radio(
            "Correlation Analysis Type",
            options=["Signal Correlation", "Performance Correlation", "Market Condition Sensitivity"],
            index=0,
            horizontal=True,
            key="corr_type"
        )
        
        # Run correlation analysis button
        if st.button("Run Correlation Analysis"):
            with st.spinner("Running correlation analysis..."):
                if correlation_type == "Signal Correlation":
                    correlation_results = CorrelationAnalyzer.run_signal_correlation(
                        selected_comp_objects,
                        selected_file
                    )
                elif correlation_type == "Performance Correlation":
                    correlation_results = CorrelationAnalyzer.run_performance_correlation(
                        selected_comp_objects,
                        selected_file
                    )
                else:  # Market Condition Sensitivity
                    correlation_results = CorrelationAnalyzer.run_market_condition_sensitivity(
                        selected_comp_objects,
                        selected_file
                    )
                
                if correlation_results:
                    CorrelationAnalyzer.display_correlation_results(correlation_results, correlation_type)
                else:
                    st.error("Correlation analysis failed")
    
    def _render_real_time_monitoring(self):
        """Render the real-time monitoring tab"""
        st.header("Real-time Monitoring")
        
        # Monitoring controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Component selection
            component_type = st.selectbox(
                "Component Type",
                options=[ct.name for ct in ComponentType],
                key="monitor_component_type"
            )
            
            # Get components of selected type
            comp_type = ComponentType[component_type]
            components = self.registry.get_components_by_type(comp_type)
            
            if not components:
                st.warning(f"No {component_type.lower().replace('_', ' ')} components found")
                return
            
            # Select components to monitor
            component_ids = [c.component_id for c in components]
            selected_components = st.multiselect(
                "Select components to monitor",
                options=component_ids,
                key="monitor_component_ids"
            )
        
        with col2:
            # Monitoring settings
            update_interval = st.number_input(
                "Update Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5,
                key="monitor_interval"
            )
            
            max_history = st.number_input(
                "History Length (points)",
                min_value=10,
                max_value=1000,
                value=100,
                key="monitor_history_length"
            )
            
            # Start/Stop monitoring button
            monitoring_active = st.session_state.get('monitoring_active', False)
            
            if monitoring_active:
                if st.button("Stop Monitoring"):
                    st.session_state.monitoring_active = False
                    
                    # Stop monitoring threads
                    for thread_id in st.session_state.monitoring_threads:
                        if thread_id in st.session_state.monitoring_threads:
                            thread = st.session_state.monitoring_threads[thread_id]
                            if hasattr(thread, 'stop'):
                                thread.stop()
                    
                    st.session_state.monitoring_threads = {}
                    st.success("Monitoring stopped")
            else:
                if st.button("Start Monitoring", disabled=not selected_components):
                    st.session_state.monitoring_active = True
                    st.session_state.monitor_start_time = datetime.datetime.now()
                    
                    # Initialize monitoring for components
                    for comp_id in selected_components:
                        if comp_id not in st.session_state.real_time_metrics:
                            st.session_state.real_time_metrics[comp_id] = {
                                'timestamps': [],
                                'signals': [],
                                'performance': {}
                            }
                    
                    st.success("Monitoring started")
        
        # Display real-time metrics if monitoring is active
        if st.session_state.get('monitoring_active', False) and selected_components:
            # Get selected component objects
            selected_comp_objects = [next((c for c in components if c.component_id == comp_id), None) for comp_id in selected_components]
            selected_comp_objects = [c for c in selected_comp_objects if c]
            
            # Data selection for monitoring
            data_files = self.performance_tracker.find_available_data()
            
            if not data_files:
                st.warning("No historical data files found")
                return
            
            # Use most recent data file for monitoring (would be live data in production)
            data_file = data_files[0]
            data = self.performance_tracker.load_historical_data(data_file)
            
            if data is None or data.empty:
                st.warning("No valid data available for monitoring")
                return
            
            # Update metrics (in a real implementation, this would be done by a background thread)
            current_time = datetime.datetime.now()
            
            # Check if we need to update (based on interval)
            last_update = st.session_state.get('last_monitor_update')
            
            if last_update is None or (current_time - last_update).total_seconds() >= update_interval:
                with st.spinner("Updating metrics..."):
                    updates = RealTimeMonitor.update_real_time_metrics(
                        selected_comp_objects, 
                        data, 
                        max_history
                    )
                    st.session_state.last_monitor_update = current_time
            
            # Display metrics
            RealTimeMonitor.display_real_time_metrics(selected_components)
