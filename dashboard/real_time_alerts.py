"""
Real-Time Alerts Panel Module

This module provides a real-time alerts panel for displaying risk alerts,
automated actions, and important notifications to enhance transparency.
"""

import streamlit as st
import pandas as pd
import numpy as np
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

# Alert severity levels and their colors
ALERT_LEVELS = {
    "info": "blue",
    "low": "green",
    "medium": "orange",
    "high": "red",
    "critical": "darkred"
}

def alert_card(alert: Dict[str, Any], key: Optional[str] = None):
    """
    Display a single alert as a styled card.
    
    Args:
        alert: Dictionary containing alert information
        key: Optional unique key for the alert
    """
    severity = alert.get('severity', 'info').lower()
    color = ALERT_LEVELS.get(severity, "blue")
    
    # Format timestamp
    timestamp = alert.get('timestamp', datetime.now())
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # Handle other timestamp formats if needed
            pass
    
    # Calculate how long ago the alert occurred
    now = datetime.now()
    if isinstance(timestamp, datetime):
        time_diff = now - timestamp
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}d ago"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}h ago"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}m ago"
        else:
            time_ago = f"{time_diff.seconds}s ago"
    else:
        time_ago = "unknown"
    
    # Create styled card
    with st.container(border=True):
        # Header with severity and timestamp
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<span style='color:{color};font-weight:bold;'>{severity.upper()}</span>", unsafe_allow_html=True)
        with col2:
            st.caption(f"{time_ago}")
        
        # Alert title
        title = alert.get('title', 'Alert')
        st.markdown(f"#### {title}")
        
        # Alert message
        message = alert.get('message', '')
        st.markdown(message)
        
        # Alert details
        if 'details' in alert and alert['details']:
            with st.expander("Details"):
                # Display as key-value pairs if it's a dictionary
                if isinstance(alert['details'], dict):
                    for k, v in alert['details'].items():
                        st.markdown(f"**{k}:** {v}")
                # Display as text if it's a string
                elif isinstance(alert['details'], str):
                    st.markdown(alert['details'])
                # For other types, display as is
                else:
                    st.write(alert['details'])
        
        # Action taken
        if 'action_taken' in alert and alert['action_taken']:
            st.markdown(f"**Action:** {alert['action_taken']}")
        
        # Action button for dismissible alerts
        if alert.get('dismissible', False) and key:
            if st.button("Dismiss", key=f"dismiss_{key}"):
                # In a real implementation, this would call an API to dismiss the alert
                st.success("Alert dismissed")
                # Force a rerun to refresh the UI
                st.rerun()

def filter_alerts(alerts_df: pd.DataFrame, 
                 min_severity: str = None, 
                 alert_types: List[str] = None,
                 max_age_hours: int = None) -> pd.DataFrame:
    """
    Filter alerts based on criteria.
    
    Args:
        alerts_df: DataFrame containing alerts
        min_severity: Minimum severity level to include
        alert_types: List of alert types to include
        max_age_hours: Maximum age of alerts in hours
        
    Returns:
        Filtered DataFrame
    """
    if alerts_df.empty:
        return alerts_df
    
    filtered_df = alerts_df.copy()
    
    # Filter by severity
    if min_severity:
        severity_levels = list(ALERT_LEVELS.keys())
        min_idx = severity_levels.index(min_severity.lower())
        severity_filter = [level for i, level in enumerate(severity_levels) if i >= min_idx]
        filtered_df = filtered_df[filtered_df['severity'].str.lower().isin(severity_filter)]
    
    # Filter by alert type
    if alert_types:
        filtered_df = filtered_df[filtered_df['type'].isin(alert_types)]
    
    # Filter by age
    if max_age_hours and 'timestamp' in filtered_df.columns:
        if filtered_df['timestamp'].dtype == 'object':
            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff_time]
    
    return filtered_df

def alerts_panel():
    """Display the real-time alerts panel."""
    section_header("Real-Time Risk Alerts", icon="🚨")
    
    # Get alerts data
    correlation_alerts = get_mongo_collection("correlation_risk_alerts")
    drawdown_alerts = get_mongo_collection("drawdown_threshold_alerts")
    risk_allocation_alerts = get_mongo_collection("risk_allocation_alerts")
    liquidity_alerts = get_mongo_collection("liquidity_risk_alerts")
    general_alerts = get_mongo_collection("system_alerts")
    
    # Convert to DataFrames
    correlation_df = pd.DataFrame(list(correlation_alerts)) if correlation_alerts else pd.DataFrame()
    drawdown_df = pd.DataFrame(list(drawdown_alerts)) if drawdown_alerts else pd.DataFrame()
    allocation_df = pd.DataFrame(list(risk_allocation_alerts)) if risk_allocation_alerts else pd.DataFrame()
    liquidity_df = pd.DataFrame(list(liquidity_alerts)) if liquidity_alerts else pd.DataFrame()
    general_df = pd.DataFrame(list(general_alerts)) if general_alerts else pd.DataFrame()
    
    # Add alert type column to each DataFrame
    if not correlation_df.empty:
        correlation_df['type'] = 'correlation'
    if not drawdown_df.empty:
        drawdown_df['type'] = 'drawdown'
    if not allocation_df.empty:
        allocation_df['type'] = 'allocation'
    if not liquidity_df.empty:
        liquidity_df['type'] = 'liquidity'
    if not general_df.empty:
        general_df['type'] = 'general'
    
    # Combine all alerts
    all_alerts = pd.concat([correlation_df, drawdown_df, allocation_df, liquidity_df, general_df], ignore_index=True)
    
    # If no alerts, display a message
    if all_alerts.empty:
        st.info("No alerts at this time.")
        return
    
    # Convert timestamps to datetime
    if 'timestamp' in all_alerts.columns:
        all_alerts['timestamp'] = pd.to_datetime(all_alerts['timestamp'])
        all_alerts = all_alerts.sort_values('timestamp', ascending=False)
    
    # Get unique alert types for filter
    alert_types = all_alerts['type'].unique().tolist() if 'type' in all_alerts.columns else []
    
    # Create filter controls
    with st.expander("Filter Alerts", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_severity = st.selectbox(
                "Minimum Severity",
                options=list(ALERT_LEVELS.keys()),
                index=0
            )
        
        with col2:
            selected_types = st.multiselect(
                "Alert Types",
                options=alert_types,
                default=alert_types
            )
        
        with col3:
            max_age = st.slider(
                "Max Age (hours)",
                min_value=1,
                max_value=168,  # 7 days
                value=24
            )
    
    # Filter alerts
    filtered_alerts = filter_alerts(all_alerts, min_severity, selected_types, max_age)
    
    # Display count
    st.markdown(f"**{len(filtered_alerts)} alerts**")
    
    # Display alerts in a scrollable container
    with st.container(height=400, border=False):
        # Display each alert
        for i, (_, alert) in enumerate(filtered_alerts.iterrows()):
            # Format alert data
            alert_data = {
                'severity': alert.get('severity', 'info'),
                'timestamp': alert.get('timestamp', datetime.now()),
                'title': alert.get('title', alert.get('type', 'Alert').title()),
                'message': alert.get('message', ''),
                'details': alert.get('details', {}),
                'action_taken': alert.get('action_taken', ''),
                'dismissible': alert.get('dismissible', False)
            }
            
            # Display alert card
            alert_card(alert_data, key=f"alert_{i}")
            
            # Add some spacing between cards
            st.markdown("")

def automated_actions_log():
    """Display log of automated actions taken by the system."""
    st.subheader("Automated Actions Log")
    
    # Get automated actions data
    actions_collection = get_mongo_collection("automated_actions")
    if not actions_collection:
        st.info("No automated actions data available yet.")
        return
    
    actions_df = pd.DataFrame(list(actions_collection))
    
    if actions_df.empty:
        st.info("No automated actions have been taken yet.")
        return
    
    # Sort by timestamp if available
    if 'timestamp' in actions_df.columns:
        actions_df['timestamp'] = pd.to_datetime(actions_df['timestamp'])
        actions_df = actions_df.sort_values('timestamp', ascending=False)
    
    # Display actions in a table
    if 'action' in actions_df.columns and 'reason' in actions_df.columns:
        # Select columns to display
        display_cols = ['timestamp', 'action', 'reason', 'status']
        display_cols = [col for col in display_cols if col in actions_df.columns]
        
        # Rename columns for display
        display_df = actions_df[display_cols].copy()
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Rename columns
        column_map = {
            'timestamp': 'Time',
            'action': 'Action',
            'reason': 'Reason',
            'status': 'Status'
        }
        display_df = display_df.rename(columns=column_map)
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
    else:
        # Display the dataframe as is
        st.dataframe(actions_df, use_container_width=True)

def real_time_dashboard():
    """Main real-time alerts dashboard."""
    alerts_panel()
    automated_actions_log()

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Real-Time Alerts - BensBot Dashboard",
        page_icon="🚨",
        layout="wide",
    )
    
    real_time_dashboard()
