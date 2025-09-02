"""
Strategy Rotation Visualization Module

This module provides visualizations for strategy rotation triggered by risk events,
displaying strategy compatibility scores, rotation history, and risk-driven allocation changes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from dashboard.theme import COLORS
from dashboard.components import (
    section_header, 
    styled_metric_card, 
    format_currency, 
    format_percent,
    format_number
)
from dashboard.api_utils import get_mongo_collection

def strategy_compatibility_scores(strategy_data: pd.DataFrame = None):
    """
    Display strategy compatibility scores for different market regimes.
    
    Args:
        strategy_data: DataFrame containing strategy compatibility scores
    """
    # If no data provided, fetch from database
    if strategy_data is None or strategy_data.empty:
        # Get strategy compatibility data
        compatibility_collection = get_mongo_collection("strategy_compatibility")
        if compatibility_collection:
            strategy_data = pd.DataFrame(list(compatibility_collection))
        else:
            strategy_data = pd.DataFrame()
    
    if strategy_data.empty:
        st.info("No strategy compatibility data available yet.")
        return
    
    # Display compatibility scores as a heatmap
    st.subheader("Strategy Compatibility by Market Regime")
    
    # Extract regime and strategy data
    if 'regimes' in strategy_data.columns and 'strategies' in strategy_data.columns and 'scores' in strategy_data.columns:
        # Data is in a specific format with regimes, strategies and scores columns
        regimes = strategy_data['regimes'].iloc[0] if isinstance(strategy_data['regimes'].iloc[0], list) else []
        strategies = strategy_data['strategies'].iloc[0] if isinstance(strategy_data['strategies'].iloc[0], list) else []
        scores = strategy_data['scores'].iloc[0] if isinstance(strategy_data['scores'].iloc[0], list) else []
        
        if regimes and strategies and scores:
            # Convert data to matrix form for heatmap
            heatmap_data = []
            for i, strategy in enumerate(strategies):
                row = []
                for j, regime in enumerate(regimes):
                    row.append(scores[i][j] if i < len(scores) and j < len(scores[i]) else 0)
                heatmap_data.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=regimes,
                y=strategies,
                colorscale='YlGnBu',
                hoverongaps=False))
            
            fig.update_layout(
                title="Strategy Compatibility Scores by Market Regime",
                xaxis_title="Market Regime",
                yaxis_title="Strategy",
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current market regime if available
            current_regime_collection = get_mongo_collection("current_market_regime")
            if current_regime_collection:
                current_regime_data = pd.DataFrame(list(current_regime_collection))
                if not current_regime_data.empty and 'current_regime' in current_regime_data.columns:
                    current_regime = current_regime_data.sort_values('timestamp', ascending=False)['current_regime'].iloc[0]
                    confidence = current_regime_data.sort_values('timestamp', ascending=False)['confidence'].iloc[0] if 'confidence' in current_regime_data.columns else None
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        styled_metric_card("Current Market Regime", current_regime.title())
                    if confidence is not None:
                        with col2:
                            styled_metric_card("Confidence", f"{confidence:.1%}", is_percent=True)
    else:
        # Assume data is already in the right format for display
        if 'strategy' in strategy_data.columns and 'regime' in strategy_data.columns and 'score' in strategy_data.columns:
            # Create pivot table
            pivot_data = strategy_data.pivot(index='strategy', columns='regime', values='score')
            
            # Create heatmap
            fig = px.imshow(
                pivot_data, 
                color_continuous_scale='YlGnBu',
                labels=dict(x="Market Regime", y="Strategy", color="Compatibility"),
                title="Strategy Compatibility Scores by Market Regime"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current market regime if available
            current_regime_collection = get_mongo_collection("current_market_regime")
            if current_regime_collection:
                current_regime_data = pd.DataFrame(list(current_regime_collection))
                if not current_regime_data.empty and 'current_regime' in current_regime_data.columns:
                    current_regime = current_regime_data.sort_values('timestamp', ascending=False)['current_regime'].iloc[0]
                    confidence = current_regime_data.sort_values('timestamp', ascending=False)['confidence'].iloc[0] if 'confidence' in current_regime_data.columns else None
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        styled_metric_card("Current Market Regime", current_regime.title())
                    if confidence is not None:
                        with col2:
                            styled_metric_card("Confidence", f"{confidence:.1%}", is_percent=True)

def strategy_allocation_history(allocation_data: pd.DataFrame = None):
    """
    Display strategy allocation history and changes triggered by risk events.
    
    Args:
        allocation_data: DataFrame containing strategy allocation history
    """
    # If no data provided, fetch from database
    if allocation_data is None or allocation_data.empty:
        # Get strategy allocation data
        allocation_collection = get_mongo_collection("strategy_allocation_history")
        if allocation_collection:
            allocation_data = pd.DataFrame(list(allocation_collection))
        else:
            allocation_data = pd.DataFrame()
    
    if allocation_data.empty:
        st.info("No strategy allocation history data available yet.")
        return
    
    # Convert timestamp to datetime if it's not
    if 'timestamp' in allocation_data.columns:
        allocation_data['timestamp'] = pd.to_datetime(allocation_data['timestamp'])
        allocation_data = allocation_data.sort_values('timestamp')
    
    # Display allocation changes over time
    st.subheader("Strategy Allocation History")
    
    # Check if data has expected format
    if 'allocations' in allocation_data.columns:
        # Create a time series of allocations
        # This assumes 'allocations' is a dictionary of strategy_id: allocation_pct
        # Convert the allocations to a DataFrame for easier plotting
        
        # Extract strategies and create columns
        all_strategies = set()
        for alloc in allocation_data['allocations']:
            if isinstance(alloc, dict):
                all_strategies.update(alloc.keys())
        
        # Create a DataFrame with allocation percentages over time
        allocation_ts = pd.DataFrame(index=allocation_data['timestamp'])
        
        for strategy in all_strategies:
            allocation_ts[strategy] = allocation_data['allocations'].apply(
                lambda x: x.get(strategy, 0) if isinstance(x, dict) else 0
            )
        
        # Plot allocation changes over time
        fig = px.area(
            allocation_ts, 
            x=allocation_ts.index, 
            y=allocation_ts.columns,
            title="Strategy Allocation Over Time",
            labels={"x": "Date", "y": "Allocation %"},
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display rotation events
        if 'trigger' in allocation_data.columns:
            rotation_events = allocation_data[allocation_data['trigger'].notna()]
            
            if not rotation_events.empty:
                st.subheader("Strategy Rotation Events")
                
                # Format for display
                rotation_table = rotation_events[['timestamp', 'trigger', 'reason']].copy()
                rotation_table['timestamp'] = rotation_table['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                rotation_table.columns = ['Date', 'Trigger', 'Reason']
                
                st.dataframe(rotation_table, use_container_width=True)
    else:
        st.warning("Allocation data is not in the expected format.")

def active_strategies_section():
    """Display the currently active strategies section."""
    st.subheader("Currently Active Strategies")
    
    # Get active strategies data
    active_strategies = get_mongo_collection("active_strategies")
    if not active_strategies:
        st.info("No active strategy data available yet.")
        return
    
    active_df = pd.DataFrame(list(active_strategies))
    
    if active_df.empty:
        st.info("No active strategies at this time.")
        return
    
    # Sort by timestamp if available
    if 'timestamp' in active_df.columns:
        active_df['timestamp'] = pd.to_datetime(active_df['timestamp'])
        active_df = active_df.sort_values('timestamp', ascending=False)
    
    # Display active strategies
    # Check if we have the expected data format
    if 'strategy_id' in active_df.columns and 'name' in active_df.columns:
        # Display each active strategy as a card
        cols = st.columns(3)
        for i, (_, strategy) in enumerate(active_df.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(strategy['name'])
                    if 'type' in strategy:
                        st.caption(f"Type: {strategy['type']}")
                    if 'allocation' in strategy:
                        st.metric("Allocation", f"{strategy['allocation']:.1%}")
                    if 'performance' in strategy:
                        st.metric("Performance", f"{strategy['performance']:.2%}")
                    if 'risk_score' in strategy:
                        # Color code risk score
                        risk_color = "green"
                        if strategy['risk_score'] > 70:
                            risk_color = "red"
                        elif strategy['risk_score'] > 50:
                            risk_color = "orange"
                        elif strategy['risk_score'] > 30:
                            risk_color = "yellow"
                        
                        st.markdown(f"Risk Score: <span style='color:{risk_color}'>{strategy['risk_score']:.1f}</span>", unsafe_allow_html=True)
    else:
        # Display simple active strategy list
        for col in active_df.columns:
            if col not in ['_id', 'timestamp']:
                st.write(f"**{col}:** {active_df[col].iloc[0]}")

def strategy_rotation_dashboard():
    """Main strategy rotation visualization dashboard."""
    section_header("Strategy Rotation Analytics", icon="🔄")
    
    # Display active strategies
    active_strategies_section()
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Compatibility Scores", "Allocation History"])
    
    with tab1:
        strategy_compatibility_scores()
    
    with tab2:
        strategy_allocation_history()

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Strategy Rotation - BensBot Dashboard",
        page_icon="🔄",
        layout="wide",
    )
    
    strategy_rotation_dashboard()
