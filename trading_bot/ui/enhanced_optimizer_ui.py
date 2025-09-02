"""
Enhanced Strategy Optimizer UI Module

Provides a professional user interface for strategy component optimization,
parameter exploration, and performance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
import uuid
from datetime import datetime, timedelta
import time
import threading

# Import our enhanced optimizers
from trading_bot.strategies.optimizer.enhanced_optimizer import BaseOptimizer, OptimizationResult
from trading_bot.strategies.optimizer.optimization_methods import (
    GridSearchOptimizer, RandomSearchOptimizer, BayesianOptimizer
)

# Import component system
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, 
    SignalGeneratorComponent, FilterComponent, 
    PositionSizerComponent, ExitManagerComponent
)
from trading_bot.strategies.components.component_registry import get_component_registry

# Import enhanced test framework for visualization reuse
from trading_bot.strategies.testing.enhanced_component_tester import EnhancedComponentTestCase

class EnhancedOptimizerUI:
    """
    Enhanced Streamlit UI for component optimizer
    
    Provides an interface for:
    - Configuring component optimization runs
    - Visualizing optimization results
    - Managing parameter spaces
    - Comparing component performance across parameters
    """
    
    def __init__(self, config=None):
        """
        Initialize the enhanced optimizer UI
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_folder = self.config.get('data_folder', 'data')
        self.results_folder = self.config.get('results_folder', 'optimization_results')
        
        # Ensure results folder exists
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Component registry
        self.registry = get_component_registry()
        
        # Optimization jobs
        if 'optimization_jobs' not in st.session_state:
            st.session_state.optimization_jobs = {}
        
        # Active optimization runs
        if 'active_optimizations' not in st.session_state:
            st.session_state.active_optimizations = {}
        
        # Selected component for optimization
        if 'selected_component' not in st.session_state:
            st.session_state.selected_component = None
            
        # Optimization results
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = {}
        
        # Default optimization parameters
        self.default_params = {
            'optimization_method': 'grid_search',
            'metric': 'total_profit',
            'max_evaluations': 100,
            'parallel_jobs': 4
        }
        
        # Available optimization metrics by component type
        self.metrics_by_type = {
            ComponentType.SIGNAL_GENERATOR: [
                'total_profit', 'win_rate', 'profit_factor', 
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
            ],
            ComponentType.FILTER: [
                'profit_improvement', 'quality_improvement', 
                'pass_rate', 'block_rate'
            ],
            ComponentType.POSITION_SIZER: [
                'total_weighted_return', 'size_efficiency'
            ],
            ComponentType.EXIT_MANAGER: [
                'total_profit', 'avg_hold_time', 'win_rate'
            ]
        }
        
    def render(self):
        """
        Render the enhanced optimizer UI
        """
        st.title("Component Optimizer")
        
        tabs = st.tabs([
            "Configure Optimization", 
            "Results Analysis", 
            "Parameter Space Explorer",
            "Optimization History"
        ])
        
        with tabs[0]:
            self._render_configuration_tab()
        
        with tabs[1]:
            self._render_results_tab()
            
        with tabs[2]:
            self._render_parameter_explorer_tab()
            
        with tabs[3]:
            self._render_history_tab()
            
        # Check and update running optimizations
        self._update_running_optimizations()
    
    def _render_configuration_tab(self):
        """
        Render the optimization configuration tab
        """
        st.header("Configure Component Optimization")
        
        # Component selection
        st.subheader("Select Component")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Component type selection
            component_type_options = [
                ("Signal Generators", ComponentType.SIGNAL_GENERATOR),
                ("Filters", ComponentType.FILTER),
                ("Position Sizers", ComponentType.POSITION_SIZER),
                ("Exit Managers", ComponentType.EXIT_MANAGER)
            ]
            
            selected_type_name = st.selectbox(
                "Component Type",
                [name for name, _ in component_type_options],
                index=0
            )
            
            # Get the selected enum value
            selected_type = next(enum_val for name, enum_val in component_type_options 
                              if name == selected_type_name)
            
            # Get components of selected type
            available_components = self.registry.get_components_by_type(selected_type)
            
            if not available_components:
                st.warning(f"No {selected_type_name} components available")
                return
            
            # Component selection
            component_names = [comp.component_id for comp in available_components]
            selected_component_name = st.selectbox(
                "Component",
                component_names,
                index=0 if component_names else None
            )
            
            # Get the selected component
            selected_component = next((comp for comp in available_components 
                                    if comp.component_id == selected_component_name), None)
            
            if not selected_component:
                st.warning("Component not found")
                return
            
            # Store selected component
            st.session_state.selected_component = selected_component
            
            # Show component description if available
            if hasattr(selected_component, 'description'):
                st.info(selected_component.description)
            
        with col2:
            # Show current component parameters
            if st.session_state.selected_component:
                st.subheader("Current Parameters")
                
                # Get parameters that can be optimized
                params = {}
                
                # Find all attributes that look like parameters
                for attr_name in dir(selected_component):
                    # Skip special attributes and methods
                    if attr_name.startswith('_') or callable(getattr(selected_component, attr_name)):
                        continue
                    
                    # Skip known non-parameter attributes
                    if attr_name in ['component_id', 'component_type', 'description']:
                        continue
                    
                    # Add to parameters
                    params[attr_name] = getattr(selected_component, attr_name)
                
                # Display parameters
                if params:
                    param_df = pd.DataFrame(
                        {'Parameter': list(params.keys()), 'Current Value': list(params.values())}
                    )
                    st.dataframe(param_df)
                else:
                    st.info("No configurable parameters found")
        
        # Parameter space configuration
        if st.session_state.selected_component:
            st.subheader("Configure Parameter Space")
            
            # Track parameters for optimization
            param_space = {}
            
            # Find all attributes that look like parameters
            for attr_name in dir(selected_component):
                # Skip special attributes and methods
                if attr_name.startswith('_') or callable(getattr(selected_component, attr_name)):
                    continue
                
                # Skip known non-parameter attributes
                if attr_name in ['component_id', 'component_type', 'description']:
                    continue
                
                # Get current value
                current_value = getattr(selected_component, attr_name)
                
                # Create parameter input based on type
                st.write(f"**{attr_name}**")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Checkbox to include in optimization
                    include = st.checkbox(f"Optimize", key=f"include_{attr_name}")
                
                with col2:
                    if include:
                        # Configure parameter space based on type
                        if isinstance(current_value, bool):
                            # Boolean parameter - just use both values
                            param_values = [True, False]
                            st.write("Values: True, False")
                        
                        elif isinstance(current_value, int):
                            # Integer parameter
                            min_val = st.number_input(
                                "Min", 
                                value=max(0, current_value - 10),
                                key=f"min_{attr_name}"
                            )
                            max_val = st.number_input(
                                "Max", 
                                value=current_value + 10,
                                key=f"max_{attr_name}"
                            )
                            step = st.number_input(
                                "Step", 
                                value=1,
                                min_value=1,
                                key=f"step_{attr_name}"
                            )
                            
                            # Generate values
                            param_values = list(range(min_val, max_val + 1, step))
                        
                        elif isinstance(current_value, float):
                            # Float parameter
                            min_val = st.number_input(
                                "Min", 
                                value=max(0.0, current_value - 1.0),
                                key=f"min_{attr_name}"
                            )
                            max_val = st.number_input(
                                "Max", 
                                value=current_value + 1.0,
                                key=f"max_{attr_name}"
                            )
                            step = st.number_input(
                                "Step", 
                                value=0.1,
                                key=f"step_{attr_name}"
                            )
                            
                            # Generate values
                            steps = int((max_val - min_val) / step) + 1
                            param_values = [min_val + i * step for i in range(steps)]
                        
                        elif isinstance(current_value, str):
                            # String parameter - use a text input for comma-separated values
                            default_values = current_value
                            values_input = st.text_input(
                                "Values (comma-separated)", 
                                value=default_values,
                                key=f"values_{attr_name}"
                            )
                            
                            # Parse values
                            param_values = [v.strip() for v in values_input.split(',')]
                        
                        else:
                            # Unsupported type
                            st.warning(f"Unsupported parameter type: {type(current_value)}")
                            param_values = [current_value]
                        
                        # Store parameter space
                        param_space[attr_name] = param_values
                
                with col3:
                    # Show current value
                    st.write(f"Current: {current_value}")
            
            # Optimization method
            st.subheader("Optimization Method")
            
            col1, col2 = st.columns(2)
            
            with col1:
                optimization_method = st.selectbox(
                    "Method",
                    ["grid_search", "random_search", "bayesian"],
                    index=0,
                    help="Grid: exhaustive search, Random: random sampling, Bayesian: efficient exploration"
                )
                
                # Available metrics for selected component type
                available_metrics = self.metrics_by_type.get(
                    selected_component.component_type, 
                    ['total_profit']
                )
                
                evaluation_metric = st.selectbox(
                    "Optimization Metric",
                    available_metrics,
                    index=0
                )
            
            with col2:
                # Method-specific parameters
                if optimization_method == "grid_search":
                    st.write("**Grid Search Parameters**")
                    max_evaluations = st.number_input(
                        "Max Evaluations",
                        min_value=10,
                        max_value=10000,
                        value=self.default_params['max_evaluations']
                    )
                    
                elif optimization_method == "random_search":
                    st.write("**Random Search Parameters**")
                    max_evaluations = st.number_input(
                        "Number of Trials",
                        min_value=10,
                        max_value=1000,
                        value=self.default_params['max_evaluations']
                    )
                    
                elif optimization_method == "bayesian":
                    st.write("**Bayesian Optimization Parameters**")
                    max_evaluations = st.number_input(
                        "Number of Trials",
                        min_value=10,
                        max_value=500,
                        value=min(100, self.default_params['max_evaluations'])
                    )
                
                # Common parameters
                parallel_jobs = st.number_input(
                    "Parallel Jobs",
                    min_value=1,
                    max_value=16,
                    value=self.default_params['parallel_jobs']
                )
            
            # Historical data selection
            st.subheader("Historical Data")
            
            # Data folder exploration
            data_paths = self._find_available_data()
            
            if not data_paths:
                st.warning("No historical data available. Please add data files to the data folder.")
                return
            
            selected_data_paths = st.multiselect(
                "Select Data Files",
                data_paths,
                default=[data_paths[0]] if data_paths else None
            )
            
            # Start optimization button
            if st.button("Start Optimization", type="primary"):
                if not param_space:
                    st.error("No parameters selected for optimization")
                    return
                
                if not selected_data_paths:
                    st.error("No data files selected")
                    return
                
                # Create optimization configuration
                optimization_config = {
                    'component_id': selected_component.component_id,
                    'component_type': selected_component.component_type.name,
                    'parameter_space': param_space,
                    'optimization_method': optimization_method,
                    'evaluation_metric': evaluation_metric,
                    'max_evaluations': max_evaluations,
                    'parallel_jobs': parallel_jobs,
                    'data_paths': selected_data_paths,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'id': str(uuid.uuid4())
                }
                
                # Save configuration
                self._save_optimization_config(optimization_config)
                
                # Start optimization in background
                self._start_optimization(optimization_config)
                
                # Show success message
                st.success(f"Optimization started! Job ID: {optimization_config['id']}")
    
    def _render_results_tab(self):
        """
        Render the results analysis tab
        """
        st.header("Optimization Results Analysis")
        
        # Get active and completed optimizations
        active_jobs = st.session_state.active_optimizations
        completed_jobs = {
            job_id: result 
            for job_id, result in st.session_state.optimization_results.items()
            if result.status == "completed"
        }
        
        if not active_jobs and not completed_jobs:
            st.info("No optimization jobs found. Start an optimization run from the Configuration tab.")
            return
        
        # Job selection
        all_jobs = {**active_jobs, **completed_jobs}
        job_options = []
        
        for job_id, job_info in all_jobs.items():
            if isinstance(job_info, dict) and 'component_id' in job_info:
                # Active job
                status = "🔄 Running"
                job_label = f"{job_info['component_id']} - {status} - {job_info['timestamp']}"
            else:
                # Completed job (OptimizationResult object)
                status = "✅ Completed"
                job_label = f"{job_info.component_id} - {status} - {job_info.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            
            job_options.append((job_label, job_id))
        
        selected_job_label = st.selectbox(
            "Select Optimization Job",
            [label for label, _ in job_options],
            index=0 if job_options else None
        )
        
        if not selected_job_label:
            return
        
        # Get selected job ID
        selected_job_id = next(job_id for label, job_id in job_options if label == selected_job_label)
        
        # Get job info
        job_info = all_jobs.get(selected_job_id)
        
        if not job_info:
            st.error("Job information not found")
            return
        
        # Display results
        if isinstance(job_info, dict) and 'component_id' in job_info:
            # Active job - show progress
            st.subheader(f"Optimization in Progress: {job_info['component_id']}")
            
            # Show configuration
            with st.expander("Optimization Configuration", expanded=True):
                st.json(job_info)
            
            # Show progress placeholder
            st.info("Optimization is running. Check back later for results.")
            
            # Progress bar placeholder
            st.progress(0.5, text="Running optimization...")
                
        else:
            # Completed job - show results
            self._display_optimization_results(job_info)
    
    def _render_parameter_explorer_tab(self):
        """
        Render the parameter explorer tab
        """
        st.header("Parameter Space Explorer")
        
        # Component selection (similar to configuration tab)
        st.subheader("Select Component")
        
        # Component type selection
        component_type_options = [
            ("Signal Generators", ComponentType.SIGNAL_GENERATOR),
            ("Filters", ComponentType.FILTER),
            ("Position Sizers", ComponentType.POSITION_SIZER),
            ("Exit Managers", ComponentType.EXIT_MANAGER)
        ]
        
        selected_type_name = st.selectbox(
            "Component Type",
            [name for name, _ in component_type_options],
            index=0,
            key="explorer_component_type"
        )
        
        # Get the selected enum value
        selected_type = next(enum_val for name, enum_val in component_type_options 
                          if name == selected_type_name)
        
        # Get components of selected type
        available_components = self.registry.get_components_by_type(selected_type)
        
        if not available_components:
            st.warning(f"No {selected_type_name} components available")
            return
        
        # Component selection
        component_names = [comp.component_id for comp in available_components]
        selected_component_name = st.selectbox(
            "Component",
            component_names,
            index=0 if component_names else None,
            key="explorer_component"
        )
        
        # Get the selected component
        selected_component = next((comp for comp in available_components 
                                if comp.component_id == selected_component_name), None)
        
        if not selected_component:
            st.warning("Component not found")
            return
        
        # Parameter selection
        st.subheader("Select Parameters to Explore")
        
        # Find all attributes that look like parameters
        params = {}
        for attr_name in dir(selected_component):
            # Skip special attributes and methods
            if attr_name.startswith('_') or callable(getattr(selected_component, attr_name)):
                continue
            
            # Skip known non-parameter attributes
            if attr_name in ['component_id', 'component_type', 'description']:
                continue
            
            # Add to parameters
            params[attr_name] = getattr(selected_component, attr_name)
        
        if not params:
            st.info("No configurable parameters found")
            return
        
        # Select 1-2 parameters to explore
        param_names = list(params.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            param1 = st.selectbox(
                "Parameter 1",
                param_names,
                index=0
            )
            
            # Generate values for parameter 1
            current_value1 = params[param1]
            
            if isinstance(current_value1, bool):
                param1_values = [True, False]
            elif isinstance(current_value1, int):
                param1_min = st.number_input("Min", value=max(0, current_value1 - 10), key="p1_min")
                param1_max = st.number_input("Max", value=current_value1 + 10, key="p1_max")
                param1_step = st.number_input("Step", value=1, min_value=1, key="p1_step")
                param1_values = list(range(param1_min, param1_max + 1, param1_step))
            elif isinstance(current_value1, float):
                param1_min = st.number_input("Min", value=max(0.0, current_value1 - 1.0), key="p1_min")
                param1_max = st.number_input("Max", value=current_value1 + 1.0, key="p1_max")
                param1_step = st.number_input("Step", value=0.1, key="p1_step")
                steps = int((param1_max - param1_min) / param1_step) + 1
                param1_values = [param1_min + i * param1_step for i in range(steps)]
            elif isinstance(current_value1, str):
                values_input = st.text_input("Values (comma-separated)", value=current_value1, key="p1_values")
                param1_values = [v.strip() for v in values_input.split(',')]
            else:
                st.warning(f"Unsupported parameter type: {type(current_value1)}")
                param1_values = [current_value1]
        
        with col2:
            # Optional second parameter
            use_param2 = st.checkbox("Explore second parameter")
            
            if use_param2 and len(param_names) > 1:
                # Filter out the first parameter
                param_names2 = [p for p in param_names if p != param1]
                
                param2 = st.selectbox(
                    "Parameter 2",
                    param_names2,
                    index=0
                )
                
                # Generate values for parameter 2
                current_value2 = params[param2]
                
                if isinstance(current_value2, bool):
                    param2_values = [True, False]
                elif isinstance(current_value2, int):
                    param2_min = st.number_input("Min", value=max(0, current_value2 - 10), key="p2_min")
                    param2_max = st.number_input("Max", value=current_value2 + 10, key="p2_max")
                    param2_step = st.number_input("Step", value=1, min_value=1, key="p2_step")
                    param2_values = list(range(param2_min, param2_max + 1, param2_step))
                elif isinstance(current_value2, float):
                    param2_min = st.number_input("Min", value=max(0.0, current_value2 - 1.0), key="p2_min")
                    param2_max = st.number_input("Max", value=current_value2 + 1.0, key="p2_max")
                    param2_step = st.number_input("Step", value=0.1, key="p2_step")
                    steps = int((param2_max - param2_min) / param2_step) + 1
                    param2_values = [param2_min + i * param2_step for i in range(steps)]
                elif isinstance(current_value2, str):
                    values_input = st.text_input("Values (comma-separated)", value=current_value2, key="p2_values")
                    param2_values = [v.strip() for v in values_input.split(',')]
                else:
                    st.warning(f"Unsupported parameter type: {type(current_value2)}")
                    param2_values = [current_value2]
            else:
                param2 = None
                param2_values = []
        
        # Metric selection
        st.subheader("Performance Metric")
        
        available_metrics = self.metrics_by_type.get(
            selected_component.component_type, 
            ['total_profit']
        )
        
        selected_metric = st.selectbox(
            "Metric to Visualize",
            available_metrics,
            index=0
        )
        
        # Historical data selection
        st.subheader("Historical Data")
        
        # Data folder exploration
        data_paths = self._find_available_data()
        
        if not data_paths:
            st.warning("No historical data available. Please add data files to the data folder.")
            return
        
        selected_data_path = st.selectbox(
            "Select Data File",
            data_paths,
            index=0 if data_paths else None
        )
        
        # Visualization button
        if st.button("Generate Parameter Performance Map", type="primary"):
            if not selected_data_path:
                st.error("No data file selected")
                return
            
            # Load data
            data = self._load_historical_data(selected_data_path)
            if data is None or data.empty:
                st.error(f"Failed to load data from {selected_data_path}")
                return
            
            symbol = os.path.basename(selected_data_path).split('.')[0]
            
            # Create visualization
            if not use_param2:
                self._visualize_1d_parameter_performance(
                    component=selected_component,
                    param_name=param1,
                    param_values=param1_values,
                    metric=selected_metric,
                    data=data,
                    symbol=symbol
                )
            else:
                self._visualize_2d_parameter_performance(
                    component=selected_component,
                    param1_name=param1,
                    param1_values=param1_values,
                    param2_name=param2,
                    param2_values=param2_values,
                    metric=selected_metric,
                    data=data,
                    symbol=symbol
                )
                
    def _render_history_tab(self):
        """
        Render the optimization history tab
        """
        st.header("Optimization History")
        
        # Get all saved optimization results
        saved_results = self._load_optimization_history()
        
        if not saved_results:
            st.info("No optimization history found. Complete an optimization run to see results here.")
            return
        
        # Group results by component type
        results_by_type = {}
        for result in saved_results:
            component_type = result.get('component_type', 'unknown')
            if component_type not in results_by_type:
                results_by_type[component_type] = []
            results_by_type[component_type].append(result)
        
        # Create tabs for each component type
        if not results_by_type:
            st.warning("No valid optimization results found")
            return
        
        type_tabs = st.tabs(list(results_by_type.keys()))
        
        for i, (component_type, results) in enumerate(results_by_type.items()):
            with type_tabs[i]:
                # Sort results by timestamp (most recent first)
                sorted_results = sorted(
                    results, 
                    key=lambda x: x.get('timestamp', ''), 
                    reverse=True
                )
                
                # Group by component
                results_by_component = {}
                for result in sorted_results:
                    component_id = result.get('component_id', 'unknown')
                    if component_id not in results_by_component:
                        results_by_component[component_id] = []
                    results_by_component[component_id].append(result)
                
                # Create expandable sections for each component
                for component_id, comp_results in results_by_component.items():
                    with st.expander(f"{component_id} ({len(comp_results)} runs)"):
                        # Create a table of results
                        table_data = []
                        for result in comp_results:
                            # Extract best metric value
                            best_metric = None
                            best_params = result.get('best_parameters', {})
                            best_performance = result.get('best_performance', {})
                            
                            if best_performance:
                                metric_name = result.get('evaluation_metric', 'total_profit')
                                best_metric = best_performance.get(metric_name, 
                                                               best_performance.get('total_profit'))
                            
                            # Create table row
                            row = {
                                'Timestamp': result.get('timestamp', ''),
                                'Method': result.get('optimization_method', ''),
                                'Metric': result.get('evaluation_metric', ''),
                                'Best Value': f"{best_metric:.4f}" if best_metric else "N/A",
                                'Evaluations': len(result.get('parameter_sets', [])),
                                'Status': result.get('status', 'unknown'),
                                'ID': result.get('id', '')
                            }
                            table_data.append(row)
                        
                        # Create dataframe
                        if table_data:
                            results_df = pd.DataFrame(table_data)
                            st.dataframe(results_df)
                            
                            # Select a result to view in detail
                            selected_id = st.selectbox(
                                "Select optimization run to view",
                                [row['ID'] for row in table_data],
                                key=f"select_{component_id}"
                            )
                            
                            # Find selected result
                            selected_result = next(
                                (r for r in comp_results if r.get('id') == selected_id),
                                None
                            )
                            
                            if selected_result:
                                # Show best parameters
                                st.subheader("Best Parameters")
                                best_params = selected_result.get('best_parameters', {})
                                
                                if best_params:
                                    params_df = pd.DataFrame({
                                        'Parameter': list(best_params.keys()),
                                        'Value': list(best_params.values())
                                    })
                                    st.dataframe(params_df)
                                    
                                    # Apply parameters button
                                    if st.button("Apply These Parameters", key=f"apply_{selected_id}"):
                                        component_id = selected_result.get('component_id')
                                        self._apply_parameters(component_id, best_params)
                                        st.success("Parameters applied to component!")
                                else:
                                    st.info("No best parameters available for this run")
                                
                                # Show performance information
                                st.subheader("Performance Metrics")
                                best_performance = selected_result.get('best_performance', {})
                                
                                if best_performance:
                                    metrics_df = pd.DataFrame({
                                        'Metric': list(best_performance.keys()),
                                        'Value': list(best_performance.values())
                                    })
                                    st.dataframe(metrics_df)
                                else:
                                    st.info("No performance metrics available for this run")
                                
                                # Visualization of optimization progress if available
                                progress_data = selected_result.get('progress', [])
                                if progress_data:
                                    st.subheader("Optimization Progress")
                                    
                                    # Convert to dataframe
                                    progress_df = pd.DataFrame(progress_data)
                                    
                                    # Plot progress
                                    fig = px.line(
                                        progress_df, 
                                        x='iteration' if 'iteration' in progress_df.columns else range(len(progress_df)),
                                        y='best_value' if 'best_value' in progress_df.columns else progress_df.columns[0],
                                        title="Optimization Progress"
                                    )
                                    st.plotly_chart(fig)

    def _display_optimization_results(self, result: OptimizationResult):
        """Display optimization results"""
        st.subheader(f"Results: {result.component_id}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimization Time", f"{result.optimization_time:.1f}s")
        
        with col2:
            st.metric("Parameter Sets", len(result.parameter_sets))
        
        with col3:
            # Get the best metric value
            best_metric = None
            if result.best_performance:
                for key, value in result.best_performance.items():
                    if key == result.optimization_method or key == 'total_profit':
                        best_metric = value
                        break
            
            if best_metric:
                st.metric("Best Metric Value", f"{best_metric:.4f}")
            else:
                st.metric("Best Metric Value", "N/A")
        
        with col4:
            st.metric("Status", result.status.capitalize())
        
        # Best parameters
        st.subheader("Best Parameters")
        
        if result.best_parameters:
            best_params_df = pd.DataFrame({
                'Parameter': list(result.best_parameters.keys()),
                'Value': list(result.best_parameters.values())
            })
            st.dataframe(best_params_df)
            
            # Apply parameters button
            if st.button("Apply These Parameters"):
                self._apply_parameters(result.component_id, result.best_parameters)
                st.success("Parameters applied to component!")
        else:
            st.warning("No best parameters found")
        
        # Parameter importance
        if result.parameter_importance:
            st.subheader("Parameter Importance")
            
            importance_df = pd.DataFrame({
                'Parameter': list(result.parameter_importance.keys()),
                'Importance': list(result.parameter_importance.values())
            }).sort_values('Importance', ascending=False)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Parameter', data=importance_df, ax=ax)
            ax.set_title("Parameter Importance")
            ax.set_xlabel("Relative Importance")
            st.pyplot(fig)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Collect all metrics
        all_metrics = {}
        
        for param_set in result.parameter_sets:
            if 'metrics' in param_set:
                for key, value in param_set['metrics'].items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Show metrics
        for metric_name, values in all_metrics.items():
            if not values or len(values) < 2:
                continue
                
            # Filter out non-numeric or invalid values
            numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            
            if not numeric_values:
                continue
                
            # Plot distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(numeric_values, kde=True, ax=ax)
            ax.set_title(f"{metric_name} Distribution")
            ax.axvline(max(numeric_values), color='r', linestyle='--', 
                      label=f"Best: {max(numeric_values):.4f}")
            ax.legend()
            st.pyplot(fig)
        
        # All parameter sets
        st.subheader("All Parameter Sets")
        
        # Convert to dataframe
        rows = []
        
        for i, param_set in enumerate(result.parameter_sets):
            row = {'set_id': i}
            
            # Add parameters
            if 'parameters' in param_set:
                for param_name, param_value in param_set['parameters'].items():
                    row[f"param_{param_name}"] = param_value
            
            # Add metrics
            if 'metrics' in param_set:
                for metric_name, metric_value in param_set['metrics'].items():
                    row[f"metric_{metric_name}"] = metric_value
            
            rows.append(row)
        
        if rows:
            results_df = pd.DataFrame(rows)
            st.dataframe(results_df)
        else:
            st.warning("No parameter sets data available")
    
    def _find_available_data(self):
        """
        Find available historical data files
        
        Returns:
            List of data file paths
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
    
    def _load_historical_data(self, file_path):
        """
        Load historical market data from file
        
        Args:
            file_path: Path to data file
            
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
    
    def _apply_parameters(self, component_id, parameters):
        """
        Apply parameters to a component
        
        Args:
            component_id: Component ID
            parameters: Dictionary of parameters
        """
        # Find component
        component = None
        for comp_type in ComponentType:
            components = self.registry.get_components_by_type(comp_type)
            component = next((c for c in components if c.component_id == component_id), None)
            if component:
                break
        
        if not component:
            st.warning(f"Component {component_id} not found")
            return
        
        # Apply parameters
        for param_name, param_value in parameters.items():
            if hasattr(component, param_name):
                setattr(component, param_name, param_value)
            else:
                st.warning(f"Parameter {param_name} not found in component {component_id}")
    
    def _visualize_1d_parameter_performance(self, component, param_name, param_values, metric, data, symbol):
        """
        Visualize performance for a single parameter
        
        Args:
            component: Component to test
            param_name: Parameter name
            param_values: List of parameter values to test
            metric: Metric to measure
            data: Historical data
            symbol: Symbol name
        """
        st.subheader(f"Performance Map for {param_name}")
        
        # Create progress bar
        progress_bar = st.progress(0, text="Evaluating parameters...")
        
        # Store original parameter value
        original_value = getattr(component, param_name)
        
        # Results
        param_results = []
        
        # Test each parameter value
        for i, value in enumerate(param_values):
            # Update progress
            progress = (i + 1) / len(param_values)
            progress_bar.progress(progress, text=f"Testing {param_name}={value}")
            
            # Set parameter value
            setattr(component, param_name, value)
            
            # Evaluate based on component type
            result = self._evaluate_component(component, data, symbol, metric)
            
            # Store result
            param_results.append({
                'value': value,
                'result': result
            })
        
        # Restore original value
        setattr(component, param_name, original_value)
        
        # Hide progress bar
        progress_bar.empty()
        
        # Create dataframe
        results_df = pd.DataFrame(param_results)
        
        # Display results
        st.write("Parameter values and metric results:")
        st.dataframe(results_df)
        
        # Create visualization
        if not results_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot parameter values vs results
            ax.plot(results_df['value'], results_df['result'], marker='o', linestyle='-')
            
            # Find best value
            best_idx = results_df['result'].idxmax()
            best_value = results_df.loc[best_idx, 'value']
            best_result = results_df.loc[best_idx, 'result']
            
            # Highlight best value
            ax.scatter([best_value], [best_result], color='red', s=100, zorder=5)
            ax.axvline(best_value, color='red', linestyle='--', alpha=0.3)
            
            # Annotations
            ax.text(best_value, best_result, f" Best: {best_value}", 
                   verticalalignment='bottom', horizontalalignment='left',
                   color='red', fontweight='bold')
            
            # Labels
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.set_title(f"Effect of {param_name} on {metric} for {component.component_id}")
            ax.grid(True, alpha=0.3)
            
            # Show plot
            st.pyplot(fig)
            
            # Recommendation
            st.success(f"Recommended value for {param_name}: {best_value} (yields {metric}={best_result:.4f})")
            
            # Apply button
            if st.button("Apply This Value"):
                setattr(component, param_name, best_value)
                st.success(f"Applied {param_name}={best_value} to {component.component_id}")
    
    def _visualize_2d_parameter_performance(self, component, param1_name, param1_values, param2_name, param2_values, metric, data, symbol):
        """
        Visualize performance for two parameters
        
        Args:
            component: Component to test
            param1_name: First parameter name
            param1_values: List of first parameter values
            param2_name: Second parameter name
            param2_values: List of second parameter values
            metric: Metric to measure
            data: Historical data
            symbol: Symbol name
        """
        st.subheader(f"Performance Map for {param1_name} and {param2_name}")
        
        # Create progress bar
        total_evaluations = len(param1_values) * len(param2_values)
        progress_bar = st.progress(0, text="Evaluating parameters...")
        
        # Store original parameter values
        original_value1 = getattr(component, param1_name)
        original_value2 = getattr(component, param2_name)
        
        # Results matrix
        results_matrix = np.zeros((len(param1_values), len(param2_values)))
        
        # Test each parameter combination
        count = 0
        for i, value1 in enumerate(param1_values):
            for j, value2 in enumerate(param2_values):
                # Update progress
                count += 1
                progress = count / total_evaluations
                progress_bar.progress(progress, text=f"Testing {param1_name}={value1}, {param2_name}={value2}")
                
                # Set parameter values
                setattr(component, param1_name, value1)
                setattr(component, param2_name, value2)
                
                # Evaluate based on component type
                result = self._evaluate_component(component, data, symbol, metric)
                
                # Store result
                results_matrix[i, j] = result
        
        # Restore original values
        setattr(component, param1_name, original_value1)
        setattr(component, param2_name, original_value2)
        
        # Hide progress bar
        progress_bar.empty()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Heatmap
        im = ax.imshow(results_matrix, cmap='viridis')
        
        # Color bar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric, rotation=-90, va="bottom")
        
        # Labels
        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(f"Effect of {param1_name} and {param2_name} on {metric}")
        
        # Ticks
        ax.set_xticks(np.arange(len(param2_values)))
        ax.set_yticks(np.arange(len(param1_values)))
        ax.set_xticklabels(param2_values)
        ax.set_yticklabels(param1_values)
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Find best value
        best_i, best_j = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
        best_value1 = param1_values[best_i]
        best_value2 = param2_values[best_j]
        best_result = results_matrix[best_i, best_j]
        
        # Highlight best value
        ax.plot(best_j, best_i, 'r*', markersize=15)
        
        # Show plot
        st.pyplot(fig)
        
        # Recommendation
        st.success(f"Recommended values:\n{param1_name}={best_value1}, {param2_name}={best_value2} (yields {metric}={best_result:.4f})")
        
        # Apply button
        if st.button("Apply These Values"):
            setattr(component, param1_name, best_value1)
            setattr(component, param2_name, best_value2)
            st.success(f"Applied {param1_name}={best_value1}, {param2_name}={best_value2} to {component.component_id}")
    
    def _evaluate_component(self, component, data, symbol, metric):
        """
        Evaluate component performance
        
        Args:
            component: Component to evaluate
            data: Historical data
            symbol: Symbol name
            metric: Metric to measure
            
        Returns:
            Metric value
        """
        try:
            # Create test case
            test_case = EnhancedComponentTestCase(component)
            test_case.with_historical_data(symbol, data)
            
            # Run test with metrics
            test_case.run_with_metrics()
            
            # Get performance metrics
            metrics = test_case.performance_metrics.get(symbol, {})
            
            # Get specified metric
            if metric in metrics:
                return metrics[metric]
            elif metric == 'profit_factor' and 'win_rate' in metrics:
                # Fallback for older components
                return metrics['win_rate']
            else:
                # Default to 0 if metric not found
                return 0.0
        except Exception as e:
            st.error(f"Error evaluating component: {e}")
            return 0.0
    
    def _start_optimization(self, config):
        """
        Start optimization in background thread
        
        Args:
            config: Optimization configuration
        """
        # Add to active optimizations
        st.session_state.active_optimizations[config['id']] = config
        
        # Start thread
        thread = threading.Thread(
            target=self._run_optimization_job,
            args=(config,)
        )
        thread.daemon = True
        thread.start()
        
        # Store thread reference
        if 'optimization_threads' not in st.session_state:
            st.session_state.optimization_threads = {}
        
        st.session_state.optimization_threads[config['id']] = thread
    
    def _run_optimization_job(self, config):
        """
        Run optimization job
        
        Args:
            config: Optimization configuration
        """
        try:
            # Get component
            component_id = config['component_id']
            component_type_str = config['component_type']
            component_type = ComponentType[component_type_str]
            
            component = self.registry.get_component(component_type, component_id)
            
            if not component:
                raise ValueError(f"Component {component_id} not found")
            
            # Create optimizer based on method
            method = config['optimization_method']
            
            if method == 'grid_search':
                optimizer = GridSearchOptimizer()
            elif method == 'random_search':
                optimizer = RandomSearchOptimizer()
            elif method == 'bayesian':
                optimizer = BayesianOptimizer()
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Configure optimizer
            optimizer.set_component(component)
            optimizer.with_metric(config['evaluation_metric'])
            optimizer.with_max_evaluations(config['max_evaluations'])
            optimizer.with_parallel_jobs(config['parallel_jobs'])
            
            # Load historical data
            for data_path in config['data_paths']:
                data = self._load_historical_data(data_path)
                if data is not None:
                    symbol = os.path.basename(data_path).split('.')[0]
                    optimizer.with_data(symbol, data)
            
            # Set parameter ranges
            optimizer.with_parameter_ranges(config['parameter_space'])
            
            # Run optimization
            result = optimizer.optimize()
            
            # Store result
            st.session_state.optimization_results[config['id']] = result
            
            # Save result to disk
            self._save_optimization_result(result, config)
            
            # Remove from active optimizations
            if config['id'] in st.session_state.active_optimizations:
                del st.session_state.active_optimizations[config['id']]
            
        except Exception as e:
            # Create error result
            result = OptimizationResult()
            result.component_id = config.get('component_id', '')
            result.component_type = ComponentType[config.get('component_type', 'SIGNAL_GENERATOR')]
            result.optimization_method = config.get('optimization_method', '')
            result.status = "failed"
            result.error_message = str(e)
            
            # Store error result
            st.session_state.optimization_results[config['id']] = result
            
            # Remove from active optimizations
            if config['id'] in st.session_state.active_optimizations:
                del st.session_state.active_optimizations[config['id']]
    
    def _update_running_optimizations(self):
        """
        Update status of running optimizations
        """
        # Check if there are any active optimizations
        if not hasattr(st.session_state, 'active_optimizations'):
            return
        
        active_ids = list(st.session_state.active_optimizations.keys())
        
        for job_id in active_ids:
            # Check if result is available
            if job_id in st.session_state.optimization_results:
                result = st.session_state.optimization_results[job_id]
                
                # If completed or failed, remove from active
                if result.status in ["completed", "failed"]:
                    if job_id in st.session_state.active_optimizations:
                        del st.session_state.active_optimizations[job_id]
    
    def _save_optimization_config(self, config):
        """
        Save optimization configuration
        
        Args:
            config: Optimization configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Save configuration to file
        file_path = os.path.join(
            self.results_folder,
            f"{config['component_id']}_{config['timestamp']}_config.json"
        )
        
        try:
            with open(file_path, 'w') as f:
                # Convert to serializable format
                json_config = config.copy()
                
                # Make sure all values are JSON serializable
                for k, v in json_config.items():
                    if isinstance(v, (pd.DataFrame, np.ndarray)):
                        json_config[k] = v.tolist()
                    elif isinstance(v, (datetime, type)):
                        json_config[k] = str(v)
                
                json.dump(json_config, f, indent=2)
        except Exception as e:
            st.error(f"Error saving configuration: {e}")
    
    def _save_optimization_result(self, result, config):
        """
        Save optimization result
        
        Args:
            result: Optimization result
            config: Original configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Convert result to dictionary
        result_dict = result.to_dict()
        
        # Add original configuration
        result_dict['original_config'] = config
        
        # Save result to file
        file_path = os.path.join(
            self.results_folder,
            f"{result.component_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_result.json"
        )
        
        try:
            with open(file_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        except Exception as e:
            st.error(f"Error saving result: {e}")
    
    def _load_optimization_history(self):
        """
        Load optimization history from disk
        
        Returns:
            List of optimization results
        """
        results = []
        
        # Check if results folder exists
        if not os.path.exists(self.results_folder):
            return results
        
        # Find all result files
        for file_name in os.listdir(self.results_folder):
            if file_name.endswith('_result.json'):
                file_path = os.path.join(self.results_folder, file_name)
                
                try:
                    with open(file_path, 'r') as f:
                        result_dict = json.load(f)
                        results.append(result_dict)
                except Exception as e:
                    st.warning(f"Error loading result file {file_name}: {e}")
        
        return results
