#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Alert Feed Component

Displays system alerts and notifications from the trading bot.
"""

import streamlit as st
import pandas as pd
import datetime

def render_alert_feed(limit=None):
    """
    Render the alert feed component
    
    Args:
        limit: Optional limit on number of alerts to display
    """
    # Get alerts from data service
    data_service = st.session_state.data_service
    
    # Add filter controls if no limit is provided (full page view)
    alert_type_filter = None
    
    if limit is None:
        # This is the full alerts page, add filters
        st.subheader("Filter Alerts")
        
        # Alert type filter
        alert_types = ["All Types", "INFO", "WARNING", "ERROR"]
        alert_type_filter = st.selectbox(
            "Filter by Alert Type",
            alert_types
        )
        
        if alert_type_filter == "All Types":
            alert_type_filter = None
    
    # Get alerts with filter applied
    alerts = data_service.get_alerts(limit=limit, alert_type=alert_type_filter)
    
    # Display alerts
    if alerts:
        # Create a container for alerts
        alert_container = st.container()
        
        with alert_container:
            # Create a styled HTML table for alerts
            alert_html = """
            <div style="max-height: 600px; overflow-y: auto;">
            <table style="width:100%; border-collapse: collapse;">
            """
            
            for alert in alerts:
                # Format timestamp
                timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Style based on alert type
                if alert['type'] == "ERROR":
                    bg_color = "#FFEBEE"  # Light red
                    icon = "🛑"
                elif alert['type'] == "WARNING":
                    bg_color = "#FFF8E1"  # Light yellow
                    icon = "⚠️"
                else:  # INFO
                    bg_color = "#E3F2FD"  # Light blue
                    icon = "ℹ️"
                
                # Add alert row
                alert_html += f"""
                <tr style="background-color: {bg_color}; border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px; white-space: nowrap;">{timestamp}</td>
                    <td style="padding: 8px; white-space: nowrap;">{icon} {alert['type']}</td>
                    <td style="padding: 8px;">{alert['message']}</td>
                    <td style="padding: 8px; white-space: nowrap;">{alert['source']}</td>
                </tr>
                """
            
            alert_html += """
            </table>
            </div>
            """
            
            st.write(alert_html, unsafe_allow_html=True)
            
        # Summary statistics if in full view
        if limit is None:
            st.subheader("Alert Summary")
            
            # Calculate statistics
            total_alerts = len(alerts)
            info_count = sum(1 for a in alerts if a['type'] == "INFO")
            warning_count = sum(1 for a in alerts if a['type'] == "WARNING")
            error_count = sum(1 for a in alerts if a['type'] == "ERROR")
            
            # Display statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Alerts", total_alerts)
            
            with col2:
                st.metric("Info", info_count)
            
            with col3:
                st.metric("Warnings", warning_count)
            
            with col4:
                st.metric("Errors", error_count)
    else:
        st.info("No alerts match the current filters" if limit is None else "No alerts available")
