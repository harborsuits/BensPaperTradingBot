"""
Modular Strategy UI Components

This module provides Streamlit UI components for interacting with the modular strategy system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import uuid

from trading_bot.strategies.modular_strategy_system import (
    ComponentType, MarketCondition, ActivationCondition
)
from trading_bot.strategies.modular_strategy import ModularStrategy
from trading_bot.strategies.components.component_registry import get_component_registry
from trading_bot.strategies.modular_strategy_integration import (
    ModularStrategyFactory, StrategyConfigGenerator
)

# Set default paths
DEFAULT_STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), '../../config/strategies/modular')
os.makedirs(DEFAULT_STRATEGIES_DIR, exist_ok=True)

class ModularStrategyUI:
    """UI manager for modular strategy creation and configuration."""
    
    def __init__(self, strategies_dir: str = DEFAULT_STRATEGIES_DIR):
        """
        Initialize UI manager
        
        Args:
            strategies_dir: Directory for strategy configuration files
        """
        self.strategies_dir = strategies_dir
        self.registry = get_component_registry()
        
        # Ensure directory exists
        os.makedirs(self.strategies_dir, exist_ok=True)
    
    def render_strategy_manager(self):
        """Render the strategy manager UI."""
        st.subheader("Modular Strategy Manager")
        
        # Create columns for actions
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Refresh Strategies", key="refresh_strategies"):
                st.experimental_rerun()
        
        with col2:
            if st.button("➕ New Strategy", key="new_strategy"):
                st.session_state.current_strategy = self._create_new_strategy()
                st.session_state.strategy_mode = "edit"
                st.experimental_rerun()
        
        with col3:
            if st.button("📦 Import Strategy", key="import_strategy"):
                st.session_state.strategy_mode = "import"
                st.experimental_rerun()
        
        # Display list of existing strategies
        strategy_files = [f for f in os.listdir(self.strategies_dir) if f.endswith('.json')]
        
        if not strategy_files:
            st.info("No strategies found. Create a new strategy to get started.")
        else:
            st.write("### Available Strategies")
            
            # Create a dataframe of strategies
            strategies_data = []
            
            for file in strategy_files:
                try:
                    config = self._load_strategy_config(os.path.join(self.strategies_dir, file))
                    if config:
                        strategies_data.append({
                            'ID': config.get('strategy_id', 'unknown'),
                            'Name': config.get('strategy_name', file),
                            'Components': self._count_components(config),
                            'Filename': file
                        })
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")
            
            if strategies_data:
                df = pd.DataFrame(strategies_data)
                
                # Show a selection table
                selected_indices = st.data_editor(
                    df,
                    column_config={
                        "ID": st.column_config.TextColumn("Strategy ID"),
                        "Name": st.column_config.TextColumn("Strategy Name"),
                        "Components": st.column_config.TextColumn("Components"),
                        "Filename": st.column_config.TextColumn("File")
                    },
                    hide_index=True,
                    key="strategies_table",
                    use_container_width=True,
                    disabled=True,
                    selection_mode="single"
                )
                
                if selected_indices:
                    selected_idx = selected_indices[0]
                    selected_file = df.iloc[selected_idx]['Filename']
                    selected_strategy_id = df.iloc[selected_idx]['ID']
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        if st.button("✏️ Edit", key="edit_strategy"):
                            st.session_state.current_strategy = self._load_strategy_config(
                                os.path.join(self.strategies_dir, selected_file)
                            )
                            st.session_state.strategy_mode = "edit"
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("🚀 Deploy", key="deploy_strategy"):
                            st.session_state.deploy_strategy_id = selected_strategy_id
                            st.experimental_rerun()
                    
                    with col3:
                        if st.button("🗑️ Delete", key="delete_strategy"):
                            # Add confirmation dialog
                            if st.session_state.get('confirm_delete') == selected_file:
                                os.remove(os.path.join(self.strategies_dir, selected_file))
                                st.success(f"Strategy {selected_file} deleted")
                                st.session_state.pop('confirm_delete', None)
                                st.experimental_rerun()
                            else:
                                st.session_state.confirm_delete = selected_file
                                st.warning(f"Click Delete again to confirm deletion of {selected_file}")
        
        # Handle different modes
        mode = st.session_state.get('strategy_mode')
        
        if mode == "edit":
            self.render_strategy_editor()
        elif mode == "import":
            self.render_strategy_importer()
        elif st.session_state.get('deploy_strategy_id'):
            self.render_strategy_deployment()
    
    def render_strategy_editor(self):
        """Render the strategy editor UI."""
        st.write("---")
        st.subheader("Strategy Editor")
        
        # Get current strategy
        strategy_config = st.session_state.get('current_strategy', {})
        
        # Basic strategy info
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_id = st.text_input(
                "Strategy ID", 
                value=strategy_config.get('strategy_id', f'modular_strategy_{uuid.uuid4().hex[:8]}'),
                key="strategy_id"
            )
            strategy_config['strategy_id'] = strategy_id
        
        with col2:
            strategy_name = st.text_input(
                "Strategy Name", 
                value=strategy_config.get('strategy_name', 'New Modular Strategy'),
                key="strategy_name"
            )
            strategy_config['strategy_name'] = strategy_name
        
        description = st.text_area(
            "Description", 
            value=strategy_config.get('description', 'A modular trading strategy'),
            key="strategy_description"
        )
        strategy_config['description'] = description
        
        # Component configuration
        st.write("### Strategy Components")
        
        # Create tabs for component types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Signal Generators", "Filters", "Position Sizers", 
            "Exit Managers", "Activation Rules"
        ])
        
        with tab1:
            self._render_component_editor(
                strategy_config, 
                "signal_generators", 
                "Signal Generators", 
                ComponentType.SIGNAL_GENERATOR.name
            )
        
        with tab2:
            self._render_component_editor(
                strategy_config, 
                "filters",
                "Filters",
                ComponentType.FILTER.name
            )
        
        with tab3:
            self._render_component_editor(
                strategy_config, 
                "position_sizers",
                "Position Sizers",
                ComponentType.POSITION_SIZER.name
            )
        
        with tab4:
            self._render_component_editor(
                strategy_config, 
                "exit_managers",
                "Exit Managers",
                ComponentType.EXIT_MANAGER.name
            )
        
        with tab5:
            self._render_activation_rules_editor(strategy_config)
        
        # Save strategy
        col1, col2 = st.columns([3, 1])
        
        with col1:
            filename = st.text_input(
                "Filename", 
                value=f"{strategy_id}.json",
                key="strategy_filename"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Save Strategy", key="save_strategy"):
                save_path = os.path.join(self.strategies_dir, filename)
                
                with open(save_path, 'w') as f:
                    json.dump(strategy_config, f, indent=2)
                
                st.success(f"Strategy saved to {save_path}")
                st.session_state.strategy_mode = None
                st.experimental_rerun()
        
        # Cancel editing
        if st.button("Cancel", key="cancel_edit"):
            st.session_state.strategy_mode = None
            st.experimental_rerun()
    
    def render_strategy_importer(self):
        """Render the strategy importer UI."""
        st.write("---")
        st.subheader("Import Strategy")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Strategy JSON", type=["json"])
        
        if uploaded_file:
            try:
                # Load strategy config
                strategy_config = json.load(uploaded_file)
                
                # Display strategy info
                st.write("### Strategy Details")
                st.write(f"**ID:** {strategy_config.get('strategy_id', 'unknown')}")
                st.write(f"**Name:** {strategy_config.get('strategy_name', 'unknown')}")
                st.write(f"**Description:** {strategy_config.get('description', 'No description')}")
                
                # Component counts
                comp_counts = self._count_components(strategy_config)
                st.write(f"**Components:** {comp_counts}")
                
                # Import options
                new_id = st.text_input(
                    "New Strategy ID (leave empty to keep original)", 
                    value="",
                    key="import_strategy_id"
                )
                
                if new_id:
                    strategy_config['strategy_id'] = new_id
                
                filename = st.text_input(
                    "Filename", 
                    value=f"{strategy_config.get('strategy_id', 'imported_strategy')}.json",
                    key="import_filename"
                )
                
                # Import button
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    if st.button("📥 Import", key="do_import"):
                        save_path = os.path.join(self.strategies_dir, filename)
                        
                        with open(save_path, 'w') as f:
                            json.dump(strategy_config, f, indent=2)
                        
                        st.success(f"Strategy imported to {save_path}")
                        st.session_state.strategy_mode = None
                        st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error importing strategy: {e}")
        
        # Cancel import
        if st.button("Cancel", key="cancel_import"):
            st.session_state.strategy_mode = None
            st.experimental_rerun()
    
    def render_strategy_deployment(self):
        """Render the strategy deployment UI."""
        st.write("---")
        st.subheader("Deploy Strategy")
        
        strategy_id = st.session_state.get('deploy_strategy_id')
        
        if not strategy_id:
            st.error("No strategy selected for deployment")
            return
        
        # Find strategy config
        strategy_config = None
        for file in os.listdir(self.strategies_dir):
            if file.endswith('.json'):
                config = self._load_strategy_config(os.path.join(self.strategies_dir, file))
                if config and config.get('strategy_id') == strategy_id:
                    strategy_config = config
                    break
        
        if not strategy_config:
            st.error(f"Strategy with ID {strategy_id} not found")
            return
        
        st.write(f"**Strategy:** {strategy_config.get('strategy_name', strategy_id)}")
        
        # Deployment options
        col1, col2 = st.columns(2)
        
        with col1:
            symbols = st.multiselect(
                "Symbols", 
                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "SPY", "QQQ", "IWM"],
                default=["AAPL", "MSFT"]
            )
        
        with col2:
            timeframes = st.multiselect(
                "Timeframes",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1D"],
                default=["1h", "4h"]
            )
        
        # Account allocation
        allocation = st.slider(
            "Account Allocation (%)", 
            min_value=1, 
            max_value=100, 
            value=10,
            step=1
        )
        
        # Deploy button
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🚀 Deploy Strategy", key="do_deploy"):
                # Here we would normally call the deployment logic
                st.success(f"Strategy {strategy_id} deployed to trade {', '.join(symbols)} on {', '.join(timeframes)} timeframes with {allocation}% allocation")
                st.session_state.pop('deploy_strategy_id', None)
                st.experimental_rerun()
        
        # Cancel deployment
        if st.button("Cancel", key="cancel_deploy"):
            st.session_state.pop('deploy_strategy_id', None)
            st.experimental_rerun()
    
    def _render_component_editor(self, strategy_config: Dict[str, Any], component_key: str, 
                               title: str, component_type: str):
        """Render the component editor for a specific component type."""
        st.write(f"#### {title}")
        
        # Initialize components structure if needed
        if 'components' not in strategy_config:
            strategy_config['components'] = {}
        
        if component_key not in strategy_config['components']:
            strategy_config['components'][component_key] = []
        
        # Display existing components
        components = strategy_config['components'][component_key]
        
        if not components:
            st.info(f"No {title.lower()} added yet.")
        else:
            # Create table of components
            for i, comp in enumerate(components):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{i+1}. {comp.get('class', 'Unknown')}**")
                
                with col2:
                    if st.button(f"Edit", key=f"edit_{component_key}_{i}"):
                        st.session_state[f'edit_{component_key}'] = i
                
                with col3:
                    if st.button(f"Remove", key=f"remove_{component_key}_{i}"):
                        components.pop(i)
                        st.experimental_rerun()
                
                # Show parameters if in edit mode
                if st.session_state.get(f'edit_{component_key}') == i:
                    with st.expander("Parameters", expanded=True):
                        # Get component metadata
                        metadata = self._get_component_metadata(comp.get('class'))
                        
                        if not metadata:
                            st.warning(f"No metadata found for {comp.get('class')}")
                            continue
                        
                        # Initialize parameters if needed
                        if 'parameters' not in comp:
                            comp['parameters'] = {}
                        
                        # Edit parameters
                        params = comp['parameters']
                        param_metadata = metadata.get('parameters', {})
                        
                        for param_name, param_info in param_metadata.items():
                            # Get current value
                            current_value = params.get(param_name, param_info.get('default'))
                            param_type = param_info.get('type', 'Any')
                            
                            # Create appropriate input based on type
                            if 'int' in param_type.lower():
                                params[param_name] = st.number_input(
                                    param_name, 
                                    value=int(current_value) if current_value is not None else 0,
                                    step=1,
                                    key=f"{component_key}_{i}_{param_name}"
                                )
                            elif 'float' in param_type.lower():
                                params[param_name] = st.number_input(
                                    param_name, 
                                    value=float(current_value) if current_value is not None else 0.0,
                                    step=0.1,
                                    format="%.2f",
                                    key=f"{component_key}_{i}_{param_name}"
                                )
                            elif 'bool' in param_type.lower():
                                params[param_name] = st.checkbox(
                                    param_name, 
                                    value=bool(current_value) if current_value is not None else False,
                                    key=f"{component_key}_{i}_{param_name}"
                                )
                            elif 'list' in param_type.lower() or 'array' in param_type.lower():
                                if current_value is None:
                                    current_value = []
                                
                                # Convert to string for editing
                                params[param_name] = self._parse_list_input(
                                    st.text_input(
                                        param_name, 
                                        value=', '.join(map(str, current_value)) if current_value else '',
                                        key=f"{component_key}_{i}_{param_name}"
                                    )
                                )
                            elif 'dict' in param_type.lower() or 'dictionary' in param_type.lower():
                                if current_value is None:
                                    current_value = {}
                                
                                # Convert to JSON string for editing
                                try:
                                    json_str = json.dumps(current_value)
                                except:
                                    json_str = '{}'
                                
                                json_input = st.text_area(
                                    param_name, 
                                    value=json_str,
                                    key=f"{component_key}_{i}_{param_name}"
                                )
                                
                                try:
                                    params[param_name] = json.loads(json_input)
                                except:
                                    st.error(f"Invalid JSON for {param_name}")
                            else:
                                params[param_name] = st.text_input(
                                    param_name, 
                                    value=str(current_value) if current_value is not None else '',
                                    key=f"{component_key}_{i}_{param_name}"
                                )
                        
                        # Done editing button
                        if st.button("Done", key=f"done_{component_key}_{i}"):
                            st.session_state.pop(f'edit_{component_key}', None)
                            st.experimental_rerun()
        
        # Add new component
        st.write("#### Add New Component")
        
        # Get available component classes
        component_classes = self._get_component_classes_by_type(component_type)
        
        if not component_classes:
            st.warning(f"No {title.lower()} components available")
            return
        
        # Select component class
        selected_class = st.selectbox(
            "Component Type", 
            component_classes,
            key=f"new_{component_key}_class"
        )
        
        # Add button
        if st.button("Add Component", key=f"add_{component_key}"):
            # Add new component
            components.append({
                'class': selected_class,
                'parameters': {}
            })
            
            # Set to edit mode
            st.session_state[f'edit_{component_key}'] = len(components) - 1
            st.experimental_rerun()
    
    def _render_activation_rules_editor(self, strategy_config: Dict[str, Any]):
        """Render the activation rules editor."""
        st.write("#### Activation Rules")
        
        # Initialize activation conditions if needed
        if 'activation_conditions' not in strategy_config:
            strategy_config['activation_conditions'] = []
        
        # Display existing rules
        conditions = strategy_config['activation_conditions']
        
        if not conditions:
            st.info("No activation rules added yet. Strategy will be always active.")
        else:
            # Create table of conditions
            for i, cond in enumerate(conditions):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{i+1}. {cond.get('type', 'Unknown')}**")
                
                with col2:
                    if st.button(f"Edit", key=f"edit_activation_{i}"):
                        st.session_state['edit_activation'] = i
                
                with col3:
                    if st.button(f"Remove", key=f"remove_activation_{i}"):
                        conditions.pop(i)
                        st.experimental_rerun()
                
                # Show parameters if in edit mode
                if st.session_state.get('edit_activation') == i:
                    with st.expander("Parameters", expanded=True):
                        # Initialize parameters if needed
                        if 'parameters' not in cond:
                            cond['parameters'] = {}
                        
                        # Edit parameters based on condition type
                        params = cond['parameters']
                        condition_type = cond.get('type')
                        
                        if condition_type == 'time':
                            # Time-based activation
                            params['time_zone'] = st.text_input(
                                "Time Zone", 
                                value=params.get('time_zone', 'America/New_York'),
                                key=f"activation_{i}_time_zone"
                            )
                            
                            params['start_time'] = st.text_input(
                                "Start Time (HH:MM)", 
                                value=params.get('start_time', '09:30'),
                                key=f"activation_{i}_start_time"
                            )
                            
                            params['end_time'] = st.text_input(
                                "End Time (HH:MM)", 
                                value=params.get('end_time', '16:00'),
                                key=f"activation_{i}_end_time"
                            )
                            
                            days = params.get('days_of_week', [0, 1, 2, 3, 4])
                            days_map = {
                                0: "Monday",
                                1: "Tuesday",
                                2: "Wednesday",
                                3: "Thursday",
                                4: "Friday",
                                5: "Saturday",
                                6: "Sunday"
                            }
                            
                            selected_days = st.multiselect(
                                "Days of Week",
                                options=list(range(7)),
                                default=days,
                                format_func=lambda x: days_map.get(x, str(x)),
                                key=f"activation_{i}_days"
                            )
                            
                            params['days_of_week'] = selected_days
                        
                        elif condition_type == 'indicator':
                            # Indicator-based activation
                            params['indicator'] = st.text_input(
                                "Indicator", 
                                value=params.get('indicator', 'rsi'),
                                key=f"activation_{i}_indicator"
                            )
                            
                            params['threshold'] = st.number_input(
                                "Threshold", 
                                value=float(params.get('threshold', 30.0)),
                                step=0.1,
                                format="%.1f",
                                key=f"activation_{i}_threshold"
                            )
                            
                            params['comparison'] = st.selectbox(
                                "Comparison", 
                                options=["greater_than", "less_than", "equal"],
                                index=0 if params.get('comparison') == "greater_than" else
                                      1 if params.get('comparison') == "less_than" else 2,
                                key=f"activation_{i}_comparison"
                            )
                        
                        elif condition_type == 'market':
                            # Market condition-based activation
                            all_conditions = [c.name for c in MarketCondition]
                            
                            selected_conditions = st.multiselect(
                                "Market Conditions",
                                options=all_conditions,
                                default=params.get('conditions', ["NORMAL"]),
                                key=f"activation_{i}_conditions"
                            )
                            
                            params['conditions'] = selected_conditions
                        
                        elif condition_type == 'performance':
                            # Performance-based activation
                            params['metric'] = st.selectbox(
                                "Performance Metric", 
                                options=["win_rate", "profit_factor", "sharpe", "drawdown"],
                                index=0,
                                key=f"activation_{i}_metric"
                            )
                            
                            params['threshold'] = st.number_input(
                                "Threshold", 
                                value=float(params.get('threshold', 0.5)),
                                step=0.01,
                                format="%.2f",
                                key=f"activation_{i}_threshold"
                            )
                            
                            params['lookback_periods'] = st.number_input(
                                "Lookback Periods", 
                                value=int(params.get('lookback_periods', 20)),
                                step=1,
                                key=f"activation_{i}_lookback"
                            )
                            
                            params['comparison'] = st.selectbox(
                                "Comparison", 
                                options=["greater_than", "less_than"],
                                index=0 if params.get('comparison', "greater_than") == "greater_than" else 1,
                                key=f"activation_{i}_comparison"
                            )
                        
                        # Done editing button
                        if st.button("Done", key=f"done_activation_{i}"):
                            st.session_state.pop('edit_activation', None)
                            st.experimental_rerun()
        
        # Add new activation rule
        st.write("#### Add New Activation Rule")
        
        # Select rule type
        rule_types = ["time", "indicator", "market", "performance"]
        selected_type = st.selectbox(
            "Rule Type", 
            rule_types,
            key="new_activation_type"
        )
        
        # Add button
        if st.button("Add Rule", key="add_activation"):
            # Add new rule
            conditions.append({
                'type': selected_type,
                'parameters': {}
            })
            
            # Set to edit mode
            st.session_state['edit_activation'] = len(conditions) - 1
            st.experimental_rerun()
    
    def _create_new_strategy(self) -> Dict[str, Any]:
        """Create a new empty strategy configuration."""
        return {
            'strategy_id': f'modular_strategy_{uuid.uuid4().hex[:8]}',
            'strategy_name': 'New Modular Strategy',
            'description': 'A modular trading strategy',
            'components': {
                'signal_generators': [],
                'filters': [],
                'position_sizers': [],
                'exit_managers': []
            },
            'activation_conditions': []
        }
    
    def _load_strategy_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a strategy configuration from a file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading strategy: {e}")
            return None
    
    def _count_components(self, config: Dict[str, Any]) -> str:
        """Count components in a strategy configuration."""
        if 'components' not in config:
            return "0 components"
        
        components = config['components']
        counts = {
            'signal_generators': len(components.get('signal_generators', [])),
            'filters': len(components.get('filters', [])),
            'position_sizers': len(components.get('position_sizers', [])),
            'exit_managers': len(components.get('exit_managers', []))
        }
        
        return f"{sum(counts.values())} components ({', '.join([f'{v} {k}' for k, v in counts.items() if v > 0])})"
    
    def _get_component_classes_by_type(self, component_type: str) -> List[str]:
        """Get component classes of a specific type."""
        classes = []
        
        for class_name, metadata in self.registry.get_all_component_metadata().items():
            if metadata.get('type') == component_type:
                classes.append(class_name)
        
        return sorted(classes)
    
    def _get_component_metadata(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a component class."""
        return self.registry.get_component_metadata(class_name)
    
    def _parse_list_input(self, input_str: str) -> List[Any]:
        """Parse a comma-separated string into a list."""
        if not input_str:
            return []
        
        # Split by comma and strip whitespace
        items = [item.strip() for item in input_str.split(',')]
        
        # Try to convert numeric values
        result = []
        for item in items:
            try:
                # Try to convert to int or float
                if '.' in item:
                    result.append(float(item))
                else:
                    result.append(int(item))
            except ValueError:
                # Keep as string if not numeric
                result.append(item)
        
        return result
