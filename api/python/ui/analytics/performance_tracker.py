import os
import json
import pandas as pd
import numpy as np
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union

from ...strategies.testing.enhanced_component_tester import EnhancedComponentTestCase
from ...component_registry import ComponentType


class PerformanceTracker:
    """
    Performance tracker for trading components with historical performance tracking
    and comparison capabilities
    """
    
    def __init__(self, data_folder, results_folder):
        """
        Initialize performance tracker
        
        Args:
            data_folder: Path to data folder
            results_folder: Path to results folder
        """
        self.data_folder = data_folder
        self.results_folder = results_folder
    
    def find_available_data(self):
        """
        Find available historical data files
        
        Returns:
            list: List of data file paths
        """
        data_files = []
        
        # Check data directory exists
        if not os.path.exists(self.data_folder):
            return data_files
        
        # Look for CSV files
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith('.csv'):
                    data_files.append(os.path.join(root, file))
        
        return data_files
    
    def load_historical_data(self, file_path, date_range=None):
        """
        Load historical market data from file
        
        Args:
            file_path: Path to data file
            date_range: Optional tuple of (start_date, end_date)
            
        Returns:
            DataFrame with market data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Convert date/timestamp column if present
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
                df.set_index(date_columns[0], inplace=True)
            
            # Filter by date range if provided
            if date_range and date_columns:
                start_date, end_date = date_range
                start_date = pd.Timestamp(start_date)
                end_date = pd.Timestamp(end_date)
                df = df.loc[start_date:end_date]
            
            # Ensure OHLCV columns exist
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns and col.upper() not in df.columns]
            
            if missing_columns:
                st.warning(f"Data file missing columns: {', '.join(missing_columns)}")
            
            # Standardize column names (lowercase)
            df.columns = [col.lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def filter_data_by_period(self, df, period):
        """
        Filter data by time period
        
        Args:
            df: DataFrame with datetime index
            period: Period string ('All', 'Last 30 Days', etc.)
            
        Returns:
            Filtered DataFrame
        """
        if period == "All" or df.empty:
            return df
        
        end_date = pd.Timestamp.now()
        
        if period == "Last 30 Days":
            start_date = end_date - pd.Timedelta(days=30)
        elif period == "Last 90 Days":
            start_date = end_date - pd.Timedelta(days=90)
        elif period == "Last Year":
            start_date = end_date - pd.Timedelta(days=365)
        else:
            return df
        
        return df.loc[start_date:end_date]
    
    def run_performance_test(self, component, data_files, test_name, edge_cases=True, 
                           error_cases=True, test_periods="All", date_range=None):
        """
        Run performance test on component
        
        Args:
            component: Component to test
            data_files: List of data file paths
            test_name: Name for test
            edge_cases: Whether to include edge case tests
            error_cases: Whether to include error/boundary tests
            test_periods: Period string for filtering data
            date_range: Optional custom date range
            
        Returns:
            Dict with test results
        """
        try:
            results = {
                'component_id': component.component_id,
                'test_name': test_name,
                'timestamp': datetime.datetime.now(),
                'parameters': self.get_component_parameters(component),
                'metrics': {},
                'data_files': data_files,
                'edge_cases': edge_cases,
                'error_cases': error_cases
            }
            
            # Create test case
            test_case = EnhancedComponentTestCase(component)
            
            # Configure test case
            if edge_cases:
                test_case.with_edge_cases()
                
            if error_cases:
                test_case.with_error_conditions()
            
            # Load data and run tests for each file
            for file_path in data_files:
                # Get symbol name from file
                symbol = os.path.basename(file_path).split('.')[0]
                
                # Load and filter data
                data = self.load_historical_data(file_path)
                
                if data is None or data.empty:
                    continue
                    
                # Filter by period if specified
                if test_periods != "All":
                    if test_periods == "Custom" and date_range:
                        # Custom date range already applied in load_historical_data
                        pass
                    else:
                        data = self.filter_data_by_period(data, test_periods)
                
                # Add data to test case
                test_case.with_historical_data(symbol, data)
            
            # Run test with metrics
            test_case.run_with_metrics()
            
            # Get performance metrics
            results['metrics'] = test_case.performance_metrics
            
            # Get edge case and error case results
            if edge_cases:
                results['edge_case_results'] = test_case.edge_case_results
                
            if error_cases:
                results['error_case_results'] = test_case.error_condition_results
            
            # Save results to history
            self.save_component_performance(component.component_id, results)
            
            return results
            
        except Exception as e:
            st.error(f"Error running performance test: {e}")
            return None
    
    def display_performance_test_results(self, results):
        """
        Display performance test results
        
        Args:
            results: Dict with test results
        """
        if not results:
            return
        
        st.subheader(f"Test Results: {results['test_name']}")
        
        # Display metrics for each symbol
        metrics = results['metrics']
        
        if not metrics:
            st.warning("No metrics data available")
            return
        
        # Create tabs for each symbol
        symbol_tabs = st.tabs(list(metrics.keys()))
        
        for i, symbol in enumerate(metrics.keys()):
            with symbol_tabs[i]:
                symbol_metrics = metrics[symbol]
                
                # Create metrics table
                metrics_df = pd.DataFrame([
                    {"Metric": metric, "Value": value}
                    for metric, value in symbol_metrics.items()
                ])
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Create radar chart for key metrics
                self.plot_radar_chart(symbol_metrics, symbol)
                
                # Display edge case results if available
                if 'edge_case_results' in results and symbol in results['edge_case_results']:
                    with st.expander("Edge Case Results"):
                        edge_results = results['edge_case_results'][symbol]
                        self.display_test_case_results(edge_results, "Edge Case")
                
                # Display error case results if available
                if 'error_case_results' in results and symbol in results['error_case_results']:
                    with st.expander("Error/Boundary Condition Results"):
                        error_results = results['error_case_results'][symbol]
                        self.display_test_case_results(error_results, "Error/Boundary")
    
    def display_test_case_results(self, results, case_type):
        """
        Display test case results
        
        Args:
            results: Test case results
            case_type: Type of test case
        """
        if not results:
            st.info(f"No {case_type.lower()} tests run")
            return
        
        # Convert to DataFrame
        rows = []
        
        for test_name, result in results.items():
            row = {
                "Test Name": test_name,
                "Status": "Passed" if result['passed'] else "Failed",
                "Description": result.get('description', ""),
                "Details": result.get('details', "")
            }
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Add styler to highlight passed/failed
        def highlight_status(val):
            color = "green" if val == "Passed" else "red"
            return f"background-color: {color}; color: white"
        
        # Display styled DataFrame
        st.dataframe(
            df.style.applymap(highlight_status, subset=["Status"]),
            use_container_width=True
        )
    
    def plot_radar_chart(self, metrics, title):
        """
        Plot radar chart for key metrics
        
        Args:
            metrics: Dict of metrics
            title: Chart title
        """
        # Select key metrics for radar chart
        radar_metrics = [
            "win_rate", "profit_factor", "sharpe_ratio", 
            "sortino_ratio", "calmar_ratio"
        ]
        
        # Filter metrics that exist
        available_metrics = [m for m in radar_metrics if m in metrics]
        
        if len(available_metrics) < 3:
            # Not enough metrics for a radar chart
            return
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {}
        
        for metric in available_metrics:
            value = metrics[metric]
            
            # Special handling for different metrics
            if metric == "win_rate":
                # Already 0-1
                normalized_metrics[metric] = value
            elif metric == "profit_factor":
                # 1.0 is breakeven, normalize to 0-1 with 3.0 as max
                normalized_metrics[metric] = min(max((value - 1) / 2, 0), 1)
            elif metric == "sharpe_ratio" or metric == "sortino_ratio":
                # 0 is breakeven, 3.0 is excellent
                normalized_metrics[metric] = min(max(value / 3, 0), 1)
            elif metric == "calmar_ratio":
                # 0 is breakeven, 5.0 is excellent
                normalized_metrics[metric] = min(max(value / 5, 0), 1)
            else:
                # Generic normalization
                normalized_metrics[metric] = min(max(value, 0), 1)
        
        # Create radar chart
        fig = go.Figure()
        
        # Add radar trace
        fig.add_trace(go.Scatterpolar(
            r=list(normalized_metrics.values()),
            theta=list(normalized_metrics.keys()),
            fill='toself',
            name=title
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Performance Metrics: {title}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display original values
        st.write("Original metric values:")
        for metric in available_metrics:
            st.write(f"- {metric}: {metrics[metric]:.4f}")
    
    def get_component_parameters(self, component):
        """
        Get component parameters
        
        Args:
            component: Component to get parameters for
            
        Returns:
            Dictionary of parameter names and values
        """
        # Get all public attributes that are not methods or internal attributes
        params = {}
        
        for attr in dir(component):
            # Skip private attributes and methods
            if attr.startswith('_') or callable(getattr(component, attr)):
                continue
                
            # Skip standard attributes
            if attr in ['component_id', 'component_type', 'description']:
                continue
                
            # Get parameter value
            value = getattr(component, attr)
            
            # Skip complex objects that aren't easily displayable
            if isinstance(value, (list, dict, set, tuple)):
                if len(str(value)) > 100:  # Skip large collections
                    value = f"{type(value).__name__} with {len(value)} items"
            
            params[attr] = value
            
        return params
    
    def save_component_performance(self, component_id, results):
        """
        Save component performance results
        
        Args:
            component_id: Component ID
            results: Dict with test results
        """
        # Convert to serializable format
        save_data = {
            'component_id': component_id,
            'timestamp': results['timestamp'].isoformat(),
            'parameters': results['parameters']
        }
        
        # Add metrics for each symbol
        for symbol, metrics in results['metrics'].items():
            for metric, value in metrics.items():
                save_data[f"{symbol}_{metric}"] = value
        
        # Add to session state
        if component_id not in st.session_state.component_history:
            st.session_state.component_history[component_id] = []
        
        st.session_state.component_history[component_id].append(save_data)
        
        # Save to disk
        self.save_performance_history(component_id)
    
    def save_performance_history(self, component_id):
        """
        Save performance history to disk
        
        Args:
            component_id: Component ID
        """
        # Create results directory if needed
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Get component history
        history = st.session_state.component_history.get(component_id, [])
        
        if not history:
            return
        
        # Save to file
        file_path = os.path.join(self.results_folder, f"{component_id}_performance_history.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            st.error(f"Error saving performance history: {e}")
    
    def load_component_performance_history(self, component_id):
        """
        Load component performance history from disk
        
        Args:
            component_id: Component ID
            
        Returns:
            List of performance records
        """
        # Check if results folder exists
        if not os.path.exists(self.results_folder):
            return []
        
        # Check if history file exists
        file_path = os.path.join(self.results_folder, f"{component_id}_performance_history.json")
        
        if not os.path.exists(file_path):
            return []
        
        # Load history
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
                return history
        except Exception as e:
            st.error(f"Error loading performance history: {e}")
            return []
    
    def run_component_comparison(self, components, data_files, metrics):
        """
        Run component comparison
        
        Args:
            components: List of components to compare
            data_files: List of data file paths
            metrics: List of metrics to compare
            
        Returns:
            Dict with comparison results
        """
        try:
            # Initialize results
            results = {
                'components': [c.component_id for c in components],
                'symbols': [],
                'metrics': metrics,
                'data': {}
            }
            
            # Run tests for each component and data file
            for file_path in data_files:
                # Get symbol name from file
                symbol = os.path.basename(file_path).split('.')[0]
                results['symbols'].append(symbol)
                
                # Load data
                data = self.load_historical_data(file_path)
                
                if data is None or data.empty:
                    continue
                
                # Initialize data for symbol
                if symbol not in results['data']:
                    results['data'][symbol] = {}
                
                # Test each component
                for component in components:
                    # Create test case
                    test_case = EnhancedComponentTestCase(component)
                    test_case.with_historical_data(symbol, data)
                    
                    # Run test with metrics
                    test_case.run_with_metrics()
                    
                    # Get performance metrics
                    component_metrics = test_case.performance_metrics.get(symbol, {})
                    
                    # Store metrics
                    results['data'][symbol][component.component_id] = {}
                    
                    for metric in metrics:
                        if metric in component_metrics:
                            results['data'][symbol][component.component_id][metric] = component_metrics[metric]
                        else:
                            results['data'][symbol][component.component_id][metric] = 0.0
            
            return results
            
        except Exception as e:
            st.error(f"Error running component comparison: {e}")
            return None
