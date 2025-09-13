"""
Broker Historical Performance Panel - Base Component

A Streamlit component for visualizing historical broker performance data,
including trend analysis, comparisons, and anomaly detection.

This file contains the basic structure and data loading components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import for type hints only, to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_bot.dashboard.services.data_service import DataService
    from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker


def format_metric_name(metric_name: str) -> str:
    """Format metric name for display"""
    # Replace underscores with spaces
    formatted = metric_name.replace('_', ' ')
    
    # Capitalize each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    # Handle special cases
    formatted = formatted.replace('Latency Mean Ms', 'Latency (ms)')
    formatted = formatted.replace('Avg Slippage Pct', 'Avg. Slippage (%)')
    formatted = formatted.replace('Avg Commission', 'Avg. Commission ($)')
    
    return formatted


def get_broker_names(data_service: 'DataService') -> Dict[str, str]:
    """Get mapping of broker IDs to display names"""
    try:
        # Get broker information from data service
        brokers = data_service.get_registered_brokers()
        
        # Create mapping of IDs to names
        return {broker["id"]: broker.get("name", broker["id"]) for broker in brokers}
        
    except Exception as e:
        st.error(f"Error getting broker names: {str(e)}")
        return {}


def get_time_range_options() -> Dict[str, timedelta]:
    """Get time range options for filtering"""
    return {
        "Last Hour": timedelta(hours=1),
        "Last 6 Hours": timedelta(hours=6),
        "Last 24 Hours": timedelta(hours=24),
        "Last 7 Days": timedelta(days=7),
        "Last 30 Days": timedelta(days=30),
        "Last 90 Days": timedelta(days=90)
    }


def render_time_filter_controls() -> Tuple[datetime, datetime]:
    """
    Render time filter controls
    
    Returns:
        Tuple of (start_time, end_time)
    """
    st.subheader("Time Range")
    
    # Time range options
    time_ranges = get_time_range_options()
    
    # Time range selector
    selected_range = st.selectbox(
        "Select Time Range",
        options=list(time_ranges.keys()),
        index=3  # Default to 7 days
    )
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - time_ranges[selected_range]
    
    # Option for custom range
    use_custom_range = st.checkbox("Use Custom Range")
    
    if use_custom_range:
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=start_time.date()
            )
            start_time = datetime.combine(start_date, datetime.min.time())
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=end_time.date()
            )
            end_time = datetime.combine(end_date, datetime.max.time())
    
    return start_time, end_time


def load_broker_performance_data(
    data_service: 'DataService',
    broker_ids: List[str],
    start_time: datetime,
    end_time: datetime,
    asset_class: Optional[str] = None,
    operation_type: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load broker performance data from data service
    
    Args:
        data_service: DataService instance
        broker_ids: List of broker IDs to load data for
        start_time: Start time for data
        end_time: End time for data
        asset_class: Optional asset class filter
        operation_type: Optional operation type filter
        
    Returns:
        Dict mapping broker_id to DataFrame of performance data
    """
    result = {}
    
    try:
        for broker_id in broker_ids:
            # Get data from data service
            data = data_service.get_broker_historical_performance(
                broker_id=broker_id,
                start_time=start_time,
                end_time=end_time,
                asset_class=asset_class,
                operation_type=operation_type
            )
            
            if not data:
                st.warning(f"No historical data found for broker {broker_id}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            result[broker_id] = df
            
    except Exception as e:
        st.error(f"Error loading broker performance data: {str(e)}")
    
    return result


def render_broker_historical_panel_base(data_service: 'DataService'):
    """
    Render the base broker historical performance panel
    
    Args:
        data_service: DataService instance for API calls
    """
    st.title("Broker Historical Performance")
    
    # Get broker information
    broker_names = get_broker_names(data_service)
    
    if not broker_names:
        st.error("No brokers found")
        return
    
    # Filter controls
    with st.expander("Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Broker selector
            selected_brokers = st.multiselect(
                "Select Brokers",
                options=list(broker_names.keys()),
                format_func=lambda x: broker_names.get(x, x),
                default=list(broker_names.keys())[:2] if len(broker_names) > 1 else list(broker_names.keys())
            )
        
        with col2:
            # Asset class and operation type
            asset_classes = ["All"] + ["equities", "forex", "options", "futures", "crypto"]
            selected_asset_class = st.selectbox(
                "Asset Class",
                options=asset_classes,
                index=0
            )
            
            operation_types = ["All"] + ["order", "quote", "data"]
            selected_operation_type = st.selectbox(
                "Operation Type",
                options=operation_types,
                index=0
            )
        
        # Time range
        start_time, end_time = render_time_filter_controls()
    
    # Convert "All" to None for API
    asset_class_filter = None if selected_asset_class == "All" else selected_asset_class
    operation_type_filter = None if selected_operation_type == "All" else selected_operation_type
    
    # Load data
    if selected_brokers:
        data = load_broker_performance_data(
            data_service=data_service,
            broker_ids=selected_brokers,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class_filter,
            operation_type=operation_type_filter
        )
        
        if not data:
            st.warning("No historical data found for the selected filters")
            return
        
        # Store in session state for other components
        st.session_state.broker_historical_data = data
        st.session_state.broker_historical_brokers = selected_brokers
        st.session_state.broker_historical_names = broker_names
        st.session_state.broker_historical_start = start_time
        st.session_state.broker_historical_end = end_time
        
        # Render placeholder for metrics to be filled by other components
        st.subheader("Performance Metrics")
        st.info("Select visualization components to see metrics")
    else:
        st.warning("Please select at least one broker")


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
    
    render_broker_historical_panel_base(MockDataService())
