import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union

from ...strategies.testing.enhanced_component_tester import EnhancedComponentTestCase
from ...component_registry import ComponentType


class CorrelationAnalyzer:
    """
    Analyzer for component correlations including signal correlation,
    performance correlation, and market condition sensitivity
    """
    
    @staticmethod
    def run_signal_correlation(components, data_file):
        """
        Run signal correlation analysis
        
        Args:
            components: List of components to analyze
            data_file: Data file path
            
        Returns:
            Dict with correlation results
        """
        try:
            # Load data
            data = pd.read_csv(data_file)
            
            # Convert date/timestamp column if present
            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                data[date_columns[0]] = pd.to_datetime(data[date_columns[0]])
                data.set_index(date_columns[0], inplace=True)
            
            # Standardize column names (lowercase)
            data.columns = [col.lower() for col in data.columns]
            
            # Get symbol name from file
            symbol = data_file.split('/')[-1].split('.')[0]
            
            # Initialize results
            results = {
                'symbol': symbol,
                'type': 'signal_correlation',
                'components': [c.component_id for c in components],
                'signals': {},
                'correlation_matrix': None,
                'timestamps': data.index.tolist()
            }
            
            # Generate signals for each component
            signals_data = {}
            
            for component in components:
                component_id = component.component_id
                
                # Create test case
                test_case = EnhancedComponentTestCase(component)
                test_case.with_historical_data(symbol, data)
                
                # Run test
                test_case.run()
                
                # Get signals data
                if component.component_type == ComponentType.SIGNAL_GENERATOR:
                    signals = test_case.get_signals(symbol)
                    signals_data[component_id] = signals
                    
                    # Store signals in results
                    results['signals'][component_id] = signals.tolist()
            
            # Create signals DataFrame
            signals_df = pd.DataFrame(signals_data, index=data.index)
            
            # Calculate correlation matrix
            correlation_matrix = signals_df.corr()
            
            # Store correlation matrix
            results['correlation_matrix'] = correlation_matrix.to_dict()
            
            return results
            
        except Exception as e:
            st.error(f"Error running signal correlation analysis: {e}")
            return None
    
    @staticmethod
    def run_performance_correlation(components, data_file):
        """
        Run performance correlation analysis
        
        Args:
            components: List of components to analyze
            data_file: Data file path
            
        Returns:
            Dict with correlation results
        """
        try:
            # Load data
            data = pd.read_csv(data_file)
            
            # Convert date/timestamp column if present
            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                data[date_columns[0]] = pd.to_datetime(data[date_columns[0]])
                data.set_index(date_columns[0], inplace=True)
            
            # Standardize column names (lowercase)
            data.columns = [col.lower() for col in data.columns]
            
            # Get symbol name from file
            symbol = data_file.split('/')[-1].split('.')[0]
            
            # Initialize results
            results = {
                'symbol': symbol,
                'type': 'performance_correlation',
                'components': [c.component_id for c in components],
                'performance_data': {},
                'correlation_matrix': None
            }
            
            # Create sliding windows for performance evaluation
            window_sizes = [30, 60, 90]  # Days
            
            # For each window size, evaluate performance
            for window_size in window_sizes:
                window_key = f"{window_size}_day"
                results['performance_data'][window_key] = {}
                
                # Skip if data is smaller than window
                if len(data) < window_size:
                    continue
                
                # Create rolling windows
                roll_data = data.rolling(window=window_size)
                
                # For each window, evaluate performance
                for start_idx in range(0, len(data) - window_size, window_size // 2):
                    end_idx = start_idx + window_size
                    
                    if end_idx > len(data):
                        end_idx = len(data)
                    
                    window_data = data.iloc[start_idx:end_idx]
                    window_period = f"{window_data.index[0].strftime('%Y-%m-%d')}_{window_data.index[-1].strftime('%Y-%m-%d')}"
                    
                    # Initialize performance metrics for this window
                    results['performance_data'][window_key][window_period] = {}
                    
                    # Evaluate each component
                    for component in components:
                        component_id = component.component_id
                        
                        # Initialize component metrics
                        if component_id not in results['performance_data'][window_key][window_period]:
                            results['performance_data'][window_key][window_period][component_id] = {}
                        
                        # Create test case
                        test_case = EnhancedComponentTestCase(component)
                        test_case.with_historical_data(symbol, window_data)
                        
                        # Run test with metrics
                        test_case.run_with_metrics()
                        
                        # Get performance metrics
                        metrics = test_case.performance_metrics.get(symbol, {})
                        
                        # Store metrics in results
                        for metric, value in metrics.items():
                            results['performance_data'][window_key][window_period][component_id][metric] = value
            
            # Calculate correlation matrix for each window size
            correlation_matrices = {}
            
            for window_size in window_sizes:
                window_key = f"{window_size}_day"
                
                if window_key not in results['performance_data']:
                    continue
                
                # Extract metrics for correlation
                metrics = ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "total_returns"]
                
                for metric in metrics:
                    # Create DataFrame with component metrics
                    metric_data = {}
                    
                    for component in components:
                        component_id = component.component_id
                        component_values = []
                        
                        for period, period_data in results['performance_data'][window_key].items():
                            if component_id in period_data and metric in period_data[component_id]:
                                component_values.append(period_data[component_id][metric])
                            else:
                                component_values.append(0.0)
                        
                        if component_values:
                            metric_data[component_id] = component_values
                    
                    # Create DataFrame
                    if metric_data:
                        metric_df = pd.DataFrame(metric_data)
                        
                        # Calculate correlation matrix
                        corr_matrix = metric_df.corr()
                        
                        # Store correlation matrix
                        if window_key not in correlation_matrices:
                            correlation_matrices[window_key] = {}
                        
                        correlation_matrices[window_key][metric] = corr_matrix.to_dict()
            
            # Store correlation matrices
            results['correlation_matrix'] = correlation_matrices
            
            return results
            
        except Exception as e:
            st.error(f"Error running performance correlation analysis: {e}")
            return None
    
    @staticmethod
    def run_market_condition_sensitivity(components, data_file):
        """
        Run market condition sensitivity analysis
        
        Args:
            components: List of components to analyze
            data_file: Data file path
            
        Returns:
            Dict with sensitivity results
        """
        try:
            # Load data
            data = pd.read_csv(data_file)
            
            # Convert date/timestamp column if present
            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                data[date_columns[0]] = pd.to_datetime(data[date_columns[0]])
                data.set_index(date_columns[0], inplace=True)
            
            # Standardize column names (lowercase)
            data.columns = [col.lower() for col in data.columns]
            
            # Get symbol name from file
            symbol = data_file.split('/')[-1].split('.')[0]
            
            # Initialize results
            results = {
                'symbol': symbol,
                'type': 'market_condition_sensitivity',
                'components': [c.component_id for c in components],
                'market_conditions': {},
                'sensitivity_data': {}
            }
            
            # Define market conditions
            market_conditions = [
                {'name': 'bullish', 'filter': lambda df: df['close'].pct_change(20) > 0.05},
                {'name': 'bearish', 'filter': lambda df: df['close'].pct_change(20) < -0.05},
                {'name': 'sideways', 'filter': lambda df: abs(df['close'].pct_change(20)) < 0.02},
                {'name': 'volatile', 'filter': lambda df: df['close'].rolling(20).std() > df['close'].rolling(60).std() * 1.5},
                {'name': 'low_volatility', 'filter': lambda df: df['close'].rolling(20).std() < df['close'].rolling(60).std() * 0.5}
            ]
            
            # Segment data by market condition
            condition_segments = {}
            
            for condition in market_conditions:
                # Apply filter
                condition_mask = condition['filter'](data)
                
                # Get data segments
                condition_data = data[condition_mask]
                
                # Store if we have enough data
                if len(condition_data) > 20:
                    condition_segments[condition['name']] = condition_data
                    
                    # Store periods
                    results['market_conditions'][condition['name']] = [
                        (condition_data.index[0].strftime('%Y-%m-%d'), condition_data.index[-1].strftime('%Y-%m-%d'))
                    ]
            
            # Evaluate components for each market condition
            for condition_name, condition_data in condition_segments.items():
                # Initialize condition results
                results['sensitivity_data'][condition_name] = {}
                
                # Evaluate each component
                for component in components:
                    component_id = component.component_id
                    
                    # Initialize component metrics
                    results['sensitivity_data'][condition_name][component_id] = {}
                    
                    # Create test case
                    test_case = EnhancedComponentTestCase(component)
                    test_case.with_historical_data(symbol, condition_data)
                    
                    # Run test with metrics
                    test_case.run_with_metrics()
                    
                    # Get performance metrics
                    metrics = test_case.performance_metrics.get(symbol, {})
                    
                    # Store metrics in results
                    for metric, value in metrics.items():
                        results['sensitivity_data'][condition_name][component_id][metric] = value
            
            return results
            
        except Exception as e:
            st.error(f"Error running market condition sensitivity analysis: {e}")
            return None
    
    @staticmethod
    def display_correlation_results(results, correlation_type):
        """
        Display correlation analysis results
        
        Args:
            results: Dict with correlation results
            correlation_type: Type of correlation analysis
        """
        if not results:
            st.warning("No correlation results available")
            return
        
        st.subheader(f"Correlation Analysis Results: {results['symbol']}")
        
        if correlation_type == "Signal Correlation":
            CorrelationAnalyzer._display_signal_correlation(results)
        elif correlation_type == "Performance Correlation":
            CorrelationAnalyzer._display_performance_correlation(results)
        else:  # Market Condition Sensitivity
            CorrelationAnalyzer._display_market_condition_sensitivity(results)
    
    @staticmethod
    def _display_signal_correlation(results):
        """
        Display signal correlation results
        
        Args:
            results: Dict with signal correlation results
        """
        if 'correlation_matrix' not in results or not results['correlation_matrix']:
            st.warning("No correlation data available")
            return
        
        # Create correlation heatmap
        correlation_df = pd.DataFrame(results['correlation_matrix'])
        
        fig = px.imshow(
            correlation_df,
            labels=dict(x="Component", y="Component", color="Correlation"),
            x=correlation_df.columns,
            y=correlation_df.columns,
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            title="Signal Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation matrix
        st.subheader("Correlation Matrix")
        st.dataframe(correlation_df, use_container_width=True)
        
        # Display signals over time
        st.subheader("Signals Over Time")
        
        if 'signals' in results and results['signals']:
            # Create signal DataFrame
            signal_df = pd.DataFrame(results['signals'], index=results['timestamps'])
            
            # Plot signals
            fig = go.Figure()
            
            for component_id in signal_df.columns:
                fig.add_trace(go.Scatter(
                    x=signal_df.index,
                    y=signal_df[component_id],
                    mode='lines',
                    name=component_id
                ))
            
            fig.update_layout(
                title="Component Signals Over Time",
                xaxis_title="Date",
                yaxis_title="Signal",
                legend_title="Components"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _display_performance_correlation(results):
        """
        Display performance correlation results
        
        Args:
            results: Dict with performance correlation results
        """
        if 'correlation_matrix' not in results or not results['correlation_matrix']:
            st.warning("No correlation data available")
            return
        
        # Create tabs for each window size
        window_tabs = st.tabs(list(results['correlation_matrix'].keys()))
        
        for i, window_key in enumerate(results['correlation_matrix'].keys()):
            with window_tabs[i]:
                st.subheader(f"Performance Correlation - {window_key.replace('_', ' ').title()}")
                
                # Create tabs for each metric
                metrics = list(results['correlation_matrix'][window_key].keys())
                metric_tabs = st.tabs([m.replace('_', ' ').title() for m in metrics])
                
                for j, metric in enumerate(metrics):
                    with metric_tabs[j]:
                        # Create correlation heatmap
                        correlation_df = pd.DataFrame(results['correlation_matrix'][window_key][metric])
                        
                        fig = px.imshow(
                            correlation_df,
                            labels=dict(x="Component", y="Component", color="Correlation"),
                            x=correlation_df.columns,
                            y=correlation_df.columns,
                            color_continuous_scale="RdBu_r",
                            range_color=[-1, 1],
                            title=f"{metric.replace('_', ' ').title()} Correlation Matrix"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display correlation matrix
                        st.dataframe(correlation_df, use_container_width=True)
    
    @staticmethod
    def _display_market_condition_sensitivity(results):
        """
        Display market condition sensitivity results
        
        Args:
            results: Dict with market condition sensitivity results
        """
        if 'sensitivity_data' not in results or not results['sensitivity_data']:
            st.warning("No market condition sensitivity data available")
            return
        
        # Create tabs for each market condition
        conditions = list(results['sensitivity_data'].keys())
        condition_tabs = st.tabs([c.replace('_', ' ').title() for c in conditions])
        
        for i, condition in enumerate(conditions):
            with condition_tabs[i]:
                st.subheader(f"Performance in {condition.replace('_', ' ').title()} Market")
                
                # Display condition periods
                if condition in results['market_conditions']:
                    periods = results['market_conditions'][condition]
                    st.write(f"Market condition periods: {', '.join([f'{start} to {end}' for start, end in periods])}")
                
                # Create metrics DataFrame
                metrics = ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "total_returns"]
                rows = []
                
                for component_id in results['components']:
                    if component_id in results['sensitivity_data'][condition]:
                        row = {"Component": component_id}
                        
                        for metric in metrics:
                            if metric in results['sensitivity_data'][condition][component_id]:
                                row[metric.replace('_', ' ').title()] = results['sensitivity_data'][condition][component_id][metric]
                            else:
                                row[metric.replace('_', ' ').title()] = 0.0
                        
                        rows.append(row)
                
                # Create DataFrame
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    
                    # Create radar chart
                    CorrelationAnalyzer._plot_market_condition_radar(df, metrics, condition)
    
    @staticmethod
    def _plot_market_condition_radar(df, metrics, condition):
        """
        Plot radar chart for market condition sensitivity
        
        Args:
            df: DataFrame with metrics
            metrics: List of metrics
            condition: Market condition name
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
                
                if max_val != min_val:
                    normalized_df[metric_title] = 1.0 - ((normalized_df[metric_title] - min_val) / (max_val - min_val))
                else:
                    normalized_df[metric_title] = 1.0
            else:
                # Higher is better for most metrics
                max_val = normalized_df[metric_title].max()
                min_val = normalized_df[metric_title].min()
                
                if max_val != min_val:
                    normalized_df[metric_title] = (normalized_df[metric_title] - min_val) / (max_val - min_val)
                else:
                    normalized_df[metric_title] = 1.0
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in normalized_df.iterrows():
            component = row['Component']
            
            # Add radar trace
            values = [row[metric.replace('_', ' ').title()] for metric in metrics if metric.replace('_', ' ').title() in row]
            
            if values:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[metric.replace('_', ' ').title() for metric in metrics if metric.replace('_', ' ').title() in row],
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
            title=f"Component Performance in {condition.replace('_', ' ').title()} Market"
        )
        
        st.plotly_chart(fig, use_container_width=True)
