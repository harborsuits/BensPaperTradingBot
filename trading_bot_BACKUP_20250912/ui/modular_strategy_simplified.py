"""
Simplified Modular Strategy UI Component

This module provides a streamlined version of the modular strategy UI that can be
safely integrated into the main app without dependency issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

class ModularStrategySimplifiedUI:
    """
    Simplified UI for modular strategy system that doesn't depend on the full components
    but demonstrates the structure and workflow.
    """
    
    def __init__(self):
        """Initialize the simplified UI components."""
        self.component_types = {
            "signal_generators": "Signal Generators",
            "filters": "Filters",
            "position_sizers": "Position Sizers",
            "exit_managers": "Exit Managers"
        }
        
        # Sample components for each type
        self.available_components = {
            "signal_generators": [
                {"id": "ma_cross", "name": "Moving Average Crossover", "description": "Generates signals when fast MA crosses slow MA"},
                {"id": "rsi", "name": "RSI", "description": "Generates signals based on RSI overbought/oversold levels"},
                {"id": "macd", "name": "MACD", "description": "Generates signals based on MACD crossovers"},
                {"id": "bb", "name": "Bollinger Bands", "description": "Generates signals when price touches/crosses bands"},
                {"id": "atr_breakout", "name": "ATR Breakout", "description": "Generates signals on volatility breakouts"}
            ],
            "filters": [
                {"id": "volume", "name": "Volume Filter", "description": "Filters signals based on volume thresholds"},
                {"id": "volatility", "name": "Volatility Filter", "description": "Filters signals based on ATR/volatility"},
                {"id": "time", "name": "Time of Day Filter", "description": "Filters signals based on market hours"},
                {"id": "trend", "name": "Trend Filter", "description": "Filters signals based on overall market trend"}
            ],
            "position_sizers": [
                {"id": "fixed_risk", "name": "Fixed Risk", "description": "Position size based on fixed risk per trade"},
                {"id": "volatility", "name": "Volatility Adjusted", "description": "Position size scaled by market volatility"},
                {"id": "kelly", "name": "Kelly Criterion", "description": "Position size based on Kelly optimal formula"},
                {"id": "equal", "name": "Equal Weight", "description": "Equal position size across all positions"}
            ],
            "exit_managers": [
                {"id": "trailing_stop", "name": "Trailing Stop", "description": "Dynamic stop loss that follows price"},
                {"id": "take_profit", "name": "Take Profit", "description": "Fixed take profit targets"},
                {"id": "time_based", "name": "Time-Based", "description": "Exit based on holding duration"},
                {"id": "technical", "name": "Technical Exit", "description": "Exit based on technical indicators"}
            ]
        }
        
        # Initialize session state for storing strategy configurations
        if "modular_strategies" not in st.session_state:
            st.session_state.modular_strategies = []
            
        if "active_strategy" not in st.session_state:
            st.session_state.active_strategy = None
            
        if "editing_strategy" not in st.session_state:
            st.session_state.editing_strategy = False
            
        # Strategy storage path
        self.strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                          "strategies", "modular_strategies")
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        # Load existing strategies
        self._load_strategies()
    
    def _load_strategies(self):
        """Load existing strategies from files."""
        if not os.path.exists(self.strategies_dir):
            return
            
        strategy_files = [f for f in os.listdir(self.strategies_dir) if f.endswith('.json')]
        
        strategies = []
        for file in strategy_files:
            try:
                with open(os.path.join(self.strategies_dir, file), 'r') as f:
                    strategy = json.load(f)
                    strategies.append(strategy)
            except Exception as e:
                st.warning(f"Failed to load strategy {file}: {e}")
                
        st.session_state.modular_strategies = strategies
    
    def _save_strategy(self, strategy):
        """Save a strategy to a file."""
        if not strategy.get("id"):
            strategy["id"] = str(uuid.uuid4())
            
        if not strategy.get("created_at"):
            strategy["created_at"] = datetime.now().isoformat()
            
        strategy["updated_at"] = datetime.now().isoformat()
        
        file_path = os.path.join(self.strategies_dir, f"{strategy['id']}.json")
        
        with open(file_path, 'w') as f:
            json.dump(strategy, f, indent=2)
            
        # Reload strategies
        self._load_strategies()
        
        return strategy
    
    def _delete_strategy(self, strategy_id):
        """Delete a strategy file."""
        file_path = os.path.join(self.strategies_dir, f"{strategy_id}.json")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Update session state
        st.session_state.modular_strategies = [s for s in st.session_state.modular_strategies 
                                              if s.get("id") != strategy_id]
    
    def _create_new_strategy(self):
        """Create a new empty strategy."""
        strategy = {
            "id": str(uuid.uuid4()),
            "name": "New Strategy",
            "description": "Strategy description",
            "signal_generators": [],
            "filters": [],
            "position_sizers": [],
            "exit_managers": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        st.session_state.active_strategy = strategy
        st.session_state.editing_strategy = True
        
        return strategy
    
    def _render_strategy_list(self):
        """Render the list of available strategies."""
        st.subheader("Available Strategies")
        
        if not st.session_state.modular_strategies:
            st.info("No strategies available. Create a new one!")
        else:
            # Create a dataframe for nice display
            strategy_data = []
            for s in st.session_state.modular_strategies:
                updated_at = datetime.fromisoformat(s.get("updated_at", s.get("created_at", "")))
                strategy_data.append({
                    "Name": s.get("name", "Unnamed"),
                    "Description": s.get("description", ""),
                    "Components": f"{len(s.get('signal_generators', []))} signals, {len(s.get('filters', []))} filters",
                    "Last Updated": updated_at.strftime("%Y-%m-%d")
                })
                
            df = pd.DataFrame(strategy_data)
            st.dataframe(df, use_container_width=True)
        
        cols = st.columns([1, 1, 1])
        
        with cols[0]:
            if st.button("New Strategy", key="new_strategy_btn"):
                self._create_new_strategy()
                
        with cols[1]:
            strategies = [s.get("name", f"Strategy {i}") for i, s in enumerate(st.session_state.modular_strategies)]
            selected = st.selectbox("Edit Strategy", ["Select..."] + strategies, key="edit_strategy_select")
            
            if selected and selected != "Select...":
                idx = strategies.index(selected)
                st.session_state.active_strategy = st.session_state.modular_strategies[idx]
                st.session_state.editing_strategy = True
                
        with cols[2]:
            strategies = [s.get("name", f"Strategy {i}") for i, s in enumerate(st.session_state.modular_strategies)]
            selected = st.selectbox("Delete Strategy", ["Select..."] + strategies, key="delete_strategy_select")
            
            if selected and selected != "Select..." and st.button("Confirm Delete", key="confirm_delete_btn"):
                idx = strategies.index(selected)
                strategy_id = st.session_state.modular_strategies[idx].get("id")
                self._delete_strategy(strategy_id)
                st.success(f"Deleted strategy: {selected}")
                st.session_state.active_strategy = None
                st.session_state.editing_strategy = False
                st.rerun()
    
    def _render_component_selector(self, component_type):
        """Render the component selector for a given component type."""
        type_name = self.component_types.get(component_type, component_type.replace("_", " ").title())
        
        st.subheader(f"{type_name}")
        
        # Get components for this type
        components = self.available_components.get(component_type, [])
        
        # Get selected components
        selected_components = st.session_state.active_strategy.get(component_type, [])
        
        # Display currently selected components
        if selected_components:
            st.write("Currently selected:")
            for comp in selected_components:
                cols = st.columns([3, 1])
                with cols[0]:
                    st.write(f"**{comp.get('name')}**: {comp.get('description')}")
                with cols[1]:
                    if st.button("Remove", key=f"remove_{component_type}_{comp.get('id')}"):
                        selected_components.remove(comp)
                        st.session_state.active_strategy[component_type] = selected_components
                        st.rerun()
        
        # Add new component
        st.write("Add new component:")
        component_names = [c.get("name") for c in components]
        selected = st.selectbox(f"Select {type_name}", ["Select..."] + component_names, key=f"select_{component_type}")
        
        if selected and selected != "Select...":
            idx = component_names.index(selected)
            component = components[idx]
            
            # Placeholder for component parameters
            st.write("Component Parameters:")
            
            # Mock parameters based on component type
            if component_type == "signal_generators":
                if component.get("id") == "ma_cross":
                    fast_period = st.number_input("Fast MA Period", min_value=1, max_value=50, value=9)
                    slow_period = st.number_input("Slow MA Period", min_value=5, max_value=200, value=21)
                    component["parameters"] = {"fast_period": fast_period, "slow_period": slow_period}
                elif component.get("id") == "rsi":
                    period = st.number_input("RSI Period", min_value=1, max_value=100, value=14)
                    overbought = st.number_input("Overbought Level", min_value=50, max_value=100, value=70)
                    oversold = st.number_input("Oversold Level", min_value=0, max_value=50, value=30)
                    component["parameters"] = {"period": period, "overbought": overbought, "oversold": oversold}
            
            elif component_type == "position_sizers":
                if component.get("id") == "fixed_risk":
                    risk_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                    max_position = st.number_input("Max Position Size (%)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
                    component["parameters"] = {"risk_pct": risk_pct, "max_position": max_position}
            
            elif component_type == "exit_managers":
                if component.get("id") == "trailing_stop":
                    initial_stop = st.number_input("Initial Stop (ATR multiplier)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
                    trail_stop = st.number_input("Trailing Stop (ATR multiplier)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
                    component["parameters"] = {"initial_stop": initial_stop, "trail_stop": trail_stop}
            
            if st.button("Add Component", key=f"add_{component_type}_{component.get('id')}"):
                selected_components.append(component)
                st.session_state.active_strategy[component_type] = selected_components
                st.success(f"Added {component.get('name')} to strategy")
                st.rerun()
    
    def _render_strategy_editor(self):
        """Render the strategy editor UI."""
        if not st.session_state.active_strategy:
            return
            
        strategy = st.session_state.active_strategy
        
        st.subheader("Strategy Configuration")
        
        # Basic info
        strategy["name"] = st.text_input("Strategy Name", value=strategy.get("name", "New Strategy"))
        strategy["description"] = st.text_area("Description", value=strategy.get("description", ""))
        
        # Component selection tabs
        component_tabs = st.tabs(list(self.component_types.values()))
        
        for i, (comp_type, _) in enumerate(self.component_types.items()):
            with component_tabs[i]:
                self._render_component_selector(comp_type)
        
        # Save and cancel buttons
        cols = st.columns([1, 1, 1])
        
        with cols[0]:
            if st.button("Save Strategy", key="save_strategy_btn"):
                self._save_strategy(strategy)
                st.success(f"Strategy '{strategy['name']}' saved successfully!")
                st.session_state.editing_strategy = False
                st.rerun()
                
        with cols[1]:
            if st.button("Cancel", key="cancel_edit_btn"):
                st.session_state.editing_strategy = False
                st.session_state.active_strategy = None
                st.rerun()
                
        with cols[2]:
            if st.button("Test Strategy", key="test_strategy_btn"):
                st.info("Strategy testing functionality will be implemented in the future.")
    
    def _render_optimization_section(self):
        """Render the strategy optimization section."""
        st.subheader("Strategy Optimization")
        
        # Select strategy to optimize
        strategies = [s.get("name", f"Strategy {i}") for i, s in enumerate(st.session_state.modular_strategies)]
        selected = st.selectbox("Select Strategy to Optimize", ["Select..."] + strategies, key="optimize_strategy_select")
        
        if not selected or selected == "Select...":
            st.info("Select a strategy to optimize")
            return
            
        # Select optimization method
        method = st.selectbox("Optimization Method", 
                             ["Grid Search", "Random Search", "Bayesian Optimization (Premium)"])
        
        # Parameters to optimize
        st.write("#### Parameters to Optimize")
        
        # Mock parameters based on selected strategy
        idx = strategies.index(selected)
        strategy = st.session_state.modular_strategies[idx]
        
        signal_generators = strategy.get("signal_generators", [])
        
        # Show parameters for each component that can be optimized
        for comp in signal_generators:
            st.write(f"**{comp.get('name')}**")
            
            if comp.get("id") == "ma_cross":
                cols = st.columns(2)
                with cols[0]:
                    st.checkbox("Fast MA Period", value=True, key=f"optimize_{comp.get('id')}_fast")
                    st.number_input("Min", value=5, key=f"optimize_{comp.get('id')}_fast_min")
                    st.number_input("Max", value=20, key=f"optimize_{comp.get('id')}_fast_max")
                    st.number_input("Step", value=1, key=f"optimize_{comp.get('id')}_fast_step")
                    
                with cols[1]:
                    st.checkbox("Slow MA Period", value=True, key=f"optimize_{comp.get('id')}_slow")
                    st.number_input("Min", value=20, key=f"optimize_{comp.get('id')}_slow_min")
                    st.number_input("Max", value=50, key=f"optimize_{comp.get('id')}_slow_max")
                    st.number_input("Step", value=5, key=f"optimize_{comp.get('id')}_slow_step")
            
            elif comp.get("id") == "rsi":
                cols = st.columns(3)
                with cols[0]:
                    st.checkbox("RSI Period", value=True, key=f"optimize_{comp.get('id')}_period")
                    st.number_input("Min", value=10, key=f"optimize_{comp.get('id')}_period_min")
                    st.number_input("Max", value=20, key=f"optimize_{comp.get('id')}_period_max")
                    st.number_input("Step", value=2, key=f"optimize_{comp.get('id')}_period_step")
        
        # Optimization settings
        st.write("#### Optimization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Optimization Goal", ["Max Profit", "Max Sharpe Ratio", "Min Drawdown", "Custom"])
            st.number_input("Max Evaluations", value=100)
            
        with col2:
            st.multiselect("Test Symbols", ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ"])
            st.date_input("Test Start Date", value=datetime(2023, 1, 1))
            st.date_input("Test End Date", value=datetime.now())
        
        # Run optimization
        if st.button("Run Optimization", key="run_optimization_btn"):
            st.info("Optimization functionality will be implemented in the future.")
            
            # Mock optimization results
            st.write("#### Optimization Results (Preview)")
            
            mock_results = pd.DataFrame({
                "Fast MA": [9, 8, 10, 7, 12],
                "Slow MA": [21, 25, 20, 30, 40],
                "Profit (%)": [15.2, 14.8, 14.1, 13.5, 13.0],
                "Sharpe": [1.8, 1.7, 1.65, 1.62, 1.58],
                "Drawdown (%)": [-8.5, -9.2, -7.8, -7.5, -10.2]
            })
            
            st.dataframe(mock_results, use_container_width=True)
            
            # Plot top results
            st.write("#### Visual Comparison")
            st.info("Performance charts will be displayed here")
    
    def _render_analytics_section(self):
        """Render the strategy analytics section."""
        st.subheader("Strategy Analytics")
        
        # Select strategy to analyze
        strategies = [s.get("name", f"Strategy {i}") for i, s in enumerate(st.session_state.modular_strategies)]
        selected = st.selectbox("Select Strategy to Analyze", ["Select..."] + strategies, key="analyze_strategy_select")
        
        if not selected or selected == "Select...":
            st.info("Select a strategy to analyze")
            return
            
        # Mock analytics results
        st.write("#### Component Performance")
        
        # Display mock component performance metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Signal Accuracy", "68.5%", "2.5%")
        with metric_cols[1]:
            st.metric("Filter Efficiency", "72.3%", "5.1%")
        with metric_cols[2]:
            st.metric("Avg Position Size", "3.2%", "-0.8%")
        with metric_cols[3]:
            st.metric("Exit Timing", "83.4%", "1.2%")
            
        # Mock component performance charts
        st.write("#### Component Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Signal Generators", "Filters", "Position Sizers", "Exit Managers"
        ])
        
        with tab1:
            st.info("Signal generation performance metrics will be displayed here")
            
            # Mock data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            
            # Mock signal performance data
            df = pd.DataFrame({
                'Date': dates,
                'Price': 100 + np.cumsum(np.random.normal(0, 1, 100)),
                'MA Signal': np.cumsum(np.random.normal(0.05, 1, 100)),
                'RSI Signal': np.cumsum(np.random.normal(0.03, 0.8, 100))
            })
            
            st.write("Signal Performance Over Time")
            st.line_chart(df.set_index('Date')[['MA Signal', 'RSI Signal']])
            
        with tab2:
            st.info("Filter performance metrics will be displayed here")
            
        with tab3:
            st.info("Position sizing performance metrics will be displayed here")
            
        with tab4:
            st.info("Exit timing performance metrics will be displayed here")
    
    def _render_marketplace_section(self):
        """Render the component marketplace section."""
        st.subheader("Component Marketplace")
        
        # Marketplace tabs
        tab1, tab2 = st.tabs(["Browse Components", "My Components"])
        
        with tab1:
            st.write("#### Available Components")
            
            # Component type filter
            comp_type = st.selectbox("Component Type", 
                                   ["All"] + list(self.component_types.values()), 
                                   key="marketplace_comp_type")
            
            # Mock search field
            st.text_input("Search Components", key="marketplace_search")
            
            # Mock component list
            if comp_type == "All" or comp_type == "Signal Generators":
                st.write("##### Signal Generators")
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write("**Adaptive MACD** - Advanced MACD with parameter adaptation")
                with cols[1]:
                    st.write("★★★★☆ (4.2)")
                with cols[2]:
                    st.button("Import", key="import_adaptive_macd")
                    
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write("**Heiken Ashi** - Candlestick pattern signal generator")
                with cols[1]:
                    st.write("★★★★★ (4.8)")
                with cols[2]:
                    st.button("Import", key="import_heiken_ashi")
            
            if comp_type == "All" or comp_type == "Filters":
                st.write("##### Filters")
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write("**Market Regime** - Filter based on overall market conditions")
                with cols[1]:
                    st.write("★★★★☆ (4.4)")
                with cols[2]:
                    st.button("Import", key="import_market_regime")
            
            if comp_type == "All" or comp_type == "Position Sizers":
                st.write("##### Position Sizers")
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write("**Adaptive Kelly** - Kelly criterion with dynamic adjustment")
                with cols[1]:
                    st.write("★★★★★ (4.7)")
                with cols[2]:
                    st.button("Import", key="import_adaptive_kelly")
            
            if comp_type == "All" or comp_type == "Exit Managers":
                st.write("##### Exit Managers")
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.write("**Multi-target** - Exit strategy with multiple targets")
                with cols[1]:
                    st.write("★★★★☆ (4.3)")
                with cols[2]:
                    st.button("Import", key="import_multi_target")
        
        with tab2:
            st.write("#### My Components")
            
            st.info("You haven't created any custom components yet.")
            
            st.button("Create New Component", key="create_component_btn")
    
    def render(self):
        """Render the UI for the modular strategy system."""
        st.header("Modular Strategy Builder")
        
        # Main tabs
        main_tabs = st.tabs([
            "Strategy Manager", "Strategy Editor", "Optimization", "Analytics", "Marketplace"
        ])
        
        with main_tabs[0]:
            self._render_strategy_list()
            
        with main_tabs[1]:
            if st.session_state.editing_strategy:
                self._render_strategy_editor()
            else:
                st.info("Select a strategy to edit or create a new one from the Strategy Manager tab.")
                
        with main_tabs[2]:
            self._render_optimization_section()
            
        with main_tabs[3]:
            self._render_analytics_section()
            
        with main_tabs[4]:
            self._render_marketplace_section()
