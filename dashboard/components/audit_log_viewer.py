"""
Audit Log Viewer Dashboard Component

This module provides a Streamlit dashboard component for viewing and analyzing
the trading system's audit log. It allows users to search, filter, and visualize
audit events for compliance, debugging, and analysis purposes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

from trading_bot.brokers.trade_audit_log import (
    TradeAuditLog, AuditEventType, 
    SqliteAuditLog, JsonFileAuditLog
)
from trading_bot.brokers.auth_manager import create_audit_log

# Configure logging
logger = logging.getLogger(__name__)


def setup_audit_log(config_path: str) -> Optional[TradeAuditLog]:
    """
    Set up audit log from configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured audit log or None if setup fails
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create audit log
        audit_log = create_audit_log(config)
        if not audit_log:
            st.error("Failed to initialize audit log. Check logs for details.")
            return None
            
        return audit_log
    except Exception as e:
        st.error(f"Error setting up audit log: {str(e)}")
        logger.error(f"Error setting up audit log: {str(e)}")
        return None


def format_event_details(details: Dict[str, Any]) -> str:
    """Format event details for display"""
    formatted = json.dumps(details, indent=2)
    return formatted


def render_audit_log_viewer(config_path: str = "config/broker_config.json"):
    """
    Render the audit log viewer dashboard
    
    Args:
        config_path: Path to the broker configuration file
    """
    st.title("Trading Audit Log")
    
    st.write("""
    This dashboard allows you to view, search, and analyze the audit log of your trading system.
    All trading operations, broker connections, and system events are recorded for compliance and debugging.
    """)
    
    # Setup audit log
    audit_log = setup_audit_log(config_path)
    if audit_log is None:
        return
    
    # Get event types for filtering
    event_types = [event_type.value for event_type in AuditEventType]
    
    # Sidebar for filters
    with st.sidebar:
        st.header("Filters")
        
        # Date range filter
        st.subheader("Date Range")
        
        # Default to last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        date_options = {
            "Last 24 hours": end_date - timedelta(days=1),
            "Last 7 days": end_date - timedelta(days=7),
            "Last 30 days": end_date - timedelta(days=30),
            "Last 90 days": end_date - timedelta(days=90),
            "All time": None
        }
        
        date_range = st.radio("Select time range", options=list(date_options.keys()))
        start_date = date_options[date_range]
        
        # Custom date range
        use_custom_dates = st.checkbox("Use custom date range")
        if use_custom_dates:
            custom_start = st.date_input("Start date", value=start_date)
            custom_end = st.date_input("End date", value=end_date)
            
            if custom_start and custom_end:
                start_date = datetime.combine(custom_start, datetime.min.time())
                end_date = datetime.combine(custom_end, datetime.max.time())
        
        # Event type filter
        st.subheader("Event Types")
        selected_event_types = st.multiselect(
            "Select event types",
            options=event_types,
            default=[]
        )
        
        # Broker filter
        st.subheader("Brokers")
        broker_filter = st.text_input("Broker ID filter (leave empty for all)")
        
        # Order filter
        st.subheader("Orders")
        order_filter = st.text_input("Order ID filter (leave empty for all)")
        
        # Strategy filter
        st.subheader("Strategies")
        strategy_filter = st.text_input("Strategy ID filter (leave empty for all)")
        
        # Refresh button
        if st.button("Refresh Data"):
            st.session_state.refresh_audit_log = True
    
    # Main content
    st.header("Audit Events")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Event Log", "Statistics", "Order History", "Visualizations"])
    
    # Query events based on filters
    try:
        query_params = {}
        
        if start_date:
            query_params["start_time"] = start_date
            
        query_params["end_time"] = end_date
        
        if selected_event_types:
            query_params["event_types"] = [AuditEventType(et) for et in selected_event_types]
            
        if broker_filter:
            query_params["broker_id"] = broker_filter
            
        if order_filter:
            query_params["order_id"] = order_filter
            
        if strategy_filter:
            query_params["strategy_id"] = strategy_filter
            
        # Query the audit log
        events = audit_log.query_events(**query_params)
        
        # Tab 1: Event Log
        with tab1:
            if not events:
                st.info("No audit events found with the current filters")
            else:
                # Convert to DataFrame for display
                events_data = []
                
                for event in events:
                    events_data.append({
                        "ID": event.event_id,
                        "Timestamp": event.timestamp,
                        "Event Type": event.event_type.value,
                        "Broker": event.broker_id or "N/A",
                        "Order ID": event.order_id or "N/A",
                        "Strategy": event.strategy_id or "N/A"
                    })
                
                df = pd.DataFrame(events_data)
                
                # Display as table
                st.dataframe(df, use_container_width=True)
                
                # Event details expander
                with st.expander("Event Details"):
                    selected_event_id = st.selectbox(
                        "Select event to view details",
                        options=[event.event_id for event in events],
                        format_func=lambda x: f"{x} - {next((e.event_type.value for e in events if e.event_id == x), '')}"
                    )
                    
                    if selected_event_id:
                        selected_event = next((e for e in events if e.event_id == selected_event_id), None)
                        if selected_event:
                            st.json(selected_event.details)
                
                # Export options
                with st.expander("Export Options"):
                    export_format = st.radio(
                        "Export format",
                        options=["CSV", "JSON", "Excel"]
                    )
                    
                    if st.button("Export Data"):
                        if export_format == "CSV":
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        elif export_format == "JSON":
                            # Create full export with details
                            full_export = []
                            for event in events:
                                event_dict = {
                                    "event_id": event.event_id,
                                    "timestamp": event.timestamp.isoformat(),
                                    "event_type": event.event_type.value,
                                    "broker_id": event.broker_id,
                                    "order_id": event.order_id,
                                    "strategy_id": event.strategy_id,
                                    "details": event.details
                                }
                                full_export.append(event_dict)
                                
                            json_str = json.dumps(full_export, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        elif export_format == "Excel":
                            # Pandas to Excel
                            output = df.to_excel(index=False)
                            st.download_button(
                                label="Download Excel",
                                data=output,
                                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
        
        # Tab 2: Statistics
        with tab2:
            if not events:
                st.info("No audit events found with the current filters")
            else:
                col1, col2 = st.columns(2)
                
                # Event type counts
                with col1:
                    st.subheader("Event Type Distribution")
                    event_type_counts = {}
                    for event in events:
                        event_type = event.event_type.value
                        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                    
                    et_df = pd.DataFrame({
                        "Event Type": list(event_type_counts.keys()),
                        "Count": list(event_type_counts.values())
                    })
                    
                    fig = px.bar(
                        et_df, 
                        x="Event Type", 
                        y="Count", 
                        color="Event Type",
                        title="Event Type Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Broker distribution
                with col2:
                    st.subheader("Broker Activity")
                    broker_counts = {}
                    for event in events:
                        if event.broker_id:
                            broker_counts[event.broker_id] = broker_counts.get(event.broker_id, 0) + 1
                    
                    if broker_counts:
                        broker_df = pd.DataFrame({
                            "Broker": list(broker_counts.keys()),
                            "Activity Count": list(broker_counts.values())
                        })
                        
                        fig = px.pie(
                            broker_df, 
                            values="Activity Count", 
                            names="Broker",
                            title="Broker Activity Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No broker-specific events found")
                
                # Events over time
                st.subheader("Events Over Time")
                
                # Group by day
                events_by_day = {}
                for event in events:
                    day = event.timestamp.date()
                    events_by_day[day] = events_by_day.get(day, 0) + 1
                
                time_df = pd.DataFrame({
                    "Date": list(events_by_day.keys()),
                    "Event Count": list(events_by_day.values())
                }).sort_values(by="Date")
                
                fig = px.line(
                    time_df, 
                    x="Date", 
                    y="Event Count",
                    title="Event Activity Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Events", len(events))
                
                with col2:
                    order_events = sum(1 for e in events if e.order_id)
                    st.metric("Order Events", order_events)
                
                with col3:
                    unique_strategies = len(set(e.strategy_id for e in events if e.strategy_id))
                    st.metric("Active Strategies", unique_strategies)
        
        # Tab 3: Order History
        with tab3:
            st.subheader("Order History")
            
            # Get unique order IDs
            order_ids = list(set(e.order_id for e in events if e.order_id))
            
            if not order_ids:
                st.info("No orders found in the current filtered events")
            else:
                # Allow selection of a specific order
                selected_order = st.selectbox(
                    "Select Order",
                    options=order_ids,
                    help="View the complete history of a specific order"
                )
                
                if selected_order:
                    # Get the order history
                    order_history = audit_log.get_order_history(selected_order)
                    
                    if not order_history:
                        st.warning(f"No history found for order {selected_order}")
                    else:
                        # Create timeline visualization
                        st.subheader(f"Timeline for Order: {selected_order}")
                        
                        # Format for timeline display
                        timeline_data = []
                        
                        for event in order_history:
                            timeline_data.append({
                                "Timestamp": event.timestamp,
                                "Event": event.event_type.value,
                                "Details": format_event_details(event.details)
                            })
                        
                        timeline_df = pd.DataFrame(timeline_data).sort_values(by="Timestamp")
                        
                        # Display as table
                        st.dataframe(timeline_df[["Timestamp", "Event"]], use_container_width=True)
                        
                        # Show order details from the most recent event
                        if timeline_df.shape[0] > 0:
                            latest_event = timeline_df.iloc[-1]
                            
                            st.subheader("Latest Order Status")
                            st.text(f"Last Event: {latest_event['Event']} at {latest_event['Timestamp']}")
                            
                            try:
                                details = json.loads(latest_event['Details'])
                                st.json(details)
                            except:
                                st.text(latest_event['Details'])
                
                # Order statistics
                st.subheader("Order Statistics")
                
                # Group orders by status
                order_status = {}
                for event in events:
                    if event.order_id and event.event_type in [
                        AuditEventType.ORDER_SUBMITTED,
                        AuditEventType.ORDER_FILLED,
                        AuditEventType.ORDER_CANCELLED,
                        AuditEventType.ORDER_REJECTED
                    ]:
                        # Consider the latest status for each order
                        order_status[event.order_id] = event.event_type
                
                # Count by status
                status_counts = {}
                for status in order_status.values():
                    status_counts[status.value] = status_counts.get(status.value, 0) + 1
                
                if status_counts:
                    status_df = pd.DataFrame({
                        "Status": list(status_counts.keys()),
                        "Count": list(status_counts.values())
                    })
                    
                    fig = px.pie(
                        status_df, 
                        values="Count", 
                        names="Status",
                        title="Order Status Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Visualizations
        with tab4:
            st.subheader("Advanced Visualizations")
            
            # Events heatmap by hour and day of week
            st.subheader("Activity Heatmap")
            
            # Process data for heatmap
            heatmap_data = {}
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for event in events:
                day_of_week = days[event.timestamp.weekday()]
                hour = event.timestamp.hour
                
                key = (day_of_week, hour)
                heatmap_data[key] = heatmap_data.get(key, 0) + 1
            
            # Create heatmap dataframe
            heatmap_rows = []
            for day in days:
                for hour in range(24):
                    count = heatmap_data.get((day, hour), 0)
                    heatmap_rows.append({
                        "Day": day,
                        "Hour": hour,
                        "Count": count
                    })
            
            heatmap_df = pd.DataFrame(heatmap_rows)
            
            # Create heatmap
            fig = px.density_heatmap(
                heatmap_df,
                x="Hour",
                y="Day",
                z="Count",
                title="Activity by Hour and Day",
                color_continuous_scale="Viridis"
            )
            
            # Customize layout
            fig.update_layout(
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
                yaxis=dict(
                    categoryorder='array',
                    categoryarray=days
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Broker performance comparison
            st.subheader("Broker Performance")
            
            # Count filled vs. rejected orders by broker
            broker_performance = {}
            
            for event in events:
                if not event.broker_id:
                    continue
                    
                if event.broker_id not in broker_performance:
                    broker_performance[event.broker_id] = {
                        "filled": 0,
                        "rejected": 0,
                        "cancelled": 0
                    }
                
                if event.event_type == AuditEventType.ORDER_FILLED:
                    broker_performance[event.broker_id]["filled"] += 1
                elif event.event_type == AuditEventType.ORDER_REJECTED:
                    broker_performance[event.broker_id]["rejected"] += 1
                elif event.event_type == AuditEventType.ORDER_CANCELLED:
                    broker_performance[event.broker_id]["cancelled"] += 1
            
            if broker_performance:
                # Create grouped bar chart
                brokers = list(broker_performance.keys())
                filled = [broker_performance[b]["filled"] for b in brokers]
                rejected = [broker_performance[b]["rejected"] for b in brokers]
                cancelled = [broker_performance[b]["cancelled"] for b in brokers]
                
                fig = go.Figure(data=[
                    go.Bar(name='Filled', x=brokers, y=filled, marker_color='green'),
                    go.Bar(name='Rejected', x=brokers, y=rejected, marker_color='red'),
                    go.Bar(name='Cancelled', x=brokers, y=cancelled, marker_color='orange')
                ])
                
                fig.update_layout(
                    title="Order Status by Broker",
                    xaxis_title="Broker",
                    yaxis_title="Count",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No broker performance data available")
    
    except Exception as e:
        st.error(f"Error querying audit log: {str(e)}")
        logger.error(f"Error querying audit log: {str(e)}")
    
    st.divider()
    st.caption("📊 Data is queried directly from the audit log and not modified")
