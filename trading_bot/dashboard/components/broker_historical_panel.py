"""
Broker Historical Performance Panel

A Streamlit component that provides comprehensive historical broker performance
visualization, trend analysis, anomaly detection, and comparison tools.

This component assembles all the subcomponents into a cohesive dashboard
interface to track broker performance over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import subcomponents
from trading_bot.dashboard.components.broker_historical_panel_base import render_broker_historical_panel_base
from trading_bot.dashboard.components.broker_historical_trend_viz import render_broker_trend_analysis
from trading_bot.dashboard.components.broker_historical_anomaly_viz import render_broker_anomaly_analysis
from trading_bot.dashboard.components.broker_historical_comparison_viz import render_broker_comparison_analysis


def render_broker_historical_panel(data_service: Any):
    """
    Render the broker historical performance panel
    
    Args:
        data_service: DataService instance for API calls
    """
    # Base component with filters and data loading
    render_broker_historical_panel_base(data_service)
    
    # Check if data is loaded
    if not hasattr(st.session_state, 'broker_historical_data') or not st.session_state.broker_historical_data:
        return
    
    # Create tabs for different analysis components
    tabs = st.tabs(["Trends & Seasonality", "Anomaly Detection", "Broker Comparison", "Data Table"])
    
    # Tab 1: Trends & Seasonality
    with tabs[0]:
        render_broker_trend_analysis(data_service)
    
    # Tab 2: Anomaly Detection
    with tabs[1]:
        render_broker_anomaly_analysis(data_service)
    
    # Tab 3: Broker Comparison
    with tabs[2]:
        render_broker_comparison_analysis(data_service)
    
    # Tab 4: Raw Data Table
    with tabs[3]:
        st.subheader("Raw Performance Data")
        
        # Get data from session state
        data = st.session_state.broker_historical_data
        broker_names = st.session_state.broker_historical_names
        
        # Broker selector for table view
        selected_broker = st.selectbox(
            "Select Broker for Data Table",
            options=list(data.keys()),
            format_func=lambda x: broker_names.get(x, x)
        )
        
        if selected_broker and selected_broker in data:
            df = data[selected_broker]
            
            # Display table with pagination
            rows_per_page = st.slider("Rows per page", 10, 100, 20)
            total_pages = max(1, len(df) // rows_per_page + (1 if len(df) % rows_per_page > 0 else 0))
            
            # Page selector
            page = st.slider("Page", 1, total_pages, 1)
            
            # Display slice of DataFrame
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(df))
            
            # Reset index for display and add datetime as column
            display_df = df.iloc[start_idx:end_idx].copy()
            display_df['timestamp'] = display_df.index
            display_df = display_df.reset_index(drop=True)
            
            # Reorder columns to put timestamp first
            cols = ['timestamp'] + [col for col in display_df.columns if col != 'timestamp']
            display_df = display_df[cols]
            
            # Display table
            st.dataframe(display_df)
            
            # Allow downloading CSV
            st.download_button(
                label="Download as CSV",
                data=df.to_csv(),
                file_name=f"{selected_broker}_performance_data.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    # For local testing
    class MockDataService:
        def get_registered_brokers(self):
            return [
                {"id": "broker_a", "name": "Broker A"},
                {"id": "broker_b", "name": "Broker B"}
            ]
        
        def get_broker_historical_performance(self, **kwargs):
            # Mock data
            return []
    
    render_broker_historical_panel(MockDataService())
