"""
Alerts Panel Component

This component displays system alerts, warnings, and notifications with color coding.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

def render_alerts_panel(data_service, max_alerts: int = 10, account_type: str = None):
    """
    Render the alerts panel.
    
    Args:
        data_service: Data service for fetching alert data
        max_alerts: Maximum number of alerts to display
        account_type: Type of account to filter alerts by
    """
    # Fetch alert data filtered by account type
    alerts_df = data_service.get_alerts(max_alerts=max_alerts, account_type=account_type)
    
    if alerts_df.empty:
        st.info("No alerts to display.")
        return
    
    # Format the alerts
    for index, alert in alerts_df.iterrows():
        # Get alert properties
        alert_type = alert.get('type', 'INFO')
        message = alert.get('message', 'Unknown alert')
        timestamp = alert.get('timestamp', datetime.now().isoformat())
        
        # Try to parse the timestamp
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
                timestamp_display = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp_display = str(timestamp)
        
        # Set color based on alert type
        if alert_type == "ERROR":
            color = "#F44336"  # Red
            icon = "🔴"
        elif alert_type == "WARNING":
            color = "#FF9800"  # Orange
            icon = "⚠️"
        elif alert_type == "SUCCESS":
            color = "#4CAF50"  # Green
            icon = "✅"
        else:  # INFO
            color = "#2196F3"  # Blue
            icon = "ℹ️"
        
        # Create the alert card
        st.markdown(f"""
        <div style="
            padding: 10px; 
            border-radius: 5px; 
            margin-bottom: 8px; 
            border-left: 4px solid {color};
            background-color: rgba({color.lstrip('#')[:2] if len(color) > 4 else color.lstrip('#')[0]+color.lstrip('#')[0], 16}, 
                                {color.lstrip('#')[2:4] if len(color) > 4 else color.lstrip('#')[1]+color.lstrip('#')[1], 16}, 
                                {color.lstrip('#')[4:6] if len(color) > 4 else color.lstrip('#')[2]+color.lstrip('#')[2], 16}, 0.1);
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: {color};">{icon} {alert_type}</span>
                <span style="color: #757575; font-size: 0.8em;">{timestamp_display}</span>
            </div>
            <div style="font-size: 0.9em;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show a "View All" button if there are more alerts
    if len(alerts_df) >= max_alerts:
        if st.button("View All Alerts"):
            st.session_state['page'] = "Alerts"
            st.rerun()
