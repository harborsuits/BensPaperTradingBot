#!/usr/bin/env python3
"""
Broker Intelligence Streamlit Component

Provides a Streamlit component for visualizing broker intelligence metrics
and recommendations for the main BensBot dashboard application.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from trading_bot.brokers.metrics.base import MetricPeriod
from trading_bot.brokers.intelligence.multi_broker_integration import BrokerIntelligenceEngine
from trading_bot.dashboard.services.data_service import DataService

# Configure logging
logger = logging.getLogger(__name__)

def render_broker_intelligence(
    data_service: DataService,
    simplified: bool = False,
    account_type: str = "Live"
):
    """
    Render broker intelligence panel in Streamlit
    
    Args:
        data_service: Data service for retrieving intelligence data
        simplified: Whether to render in simplified mode
        account_type: Account type filter (Live/Paper)
    """
    try:
        # CSS for styling
        st.markdown("""
        <style>
        /* Status badges */
        .status-badge {
            padding: 6px 12px;
            border-radius: 16px;
            font-weight: bold;
            display: inline-block;
        }
        .status-normal {
            background-color: #28a745;
            color: white;
        }
        .status-caution {
            background-color: #ffc107;
            color: black;
        }
        .status-critical {
            background-color: #dc3545;
            color: white;
        }
        
        /* Circuit breaker alert styling */
        .circuit-breaker-alert {
            margin-bottom: 15px;
            border-left: 5px solid #dc3545;
            padding: 10px 15px;
            background-color: rgba(220, 53, 69, 0.1);
        }
        
        /* Recommendation card styling */
        .recommendation-card {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .primary-recommendation {
            border-left: 5px solid #007bff;
        }
        .backup-recommendation {
            border-left: 5px solid #6c757d;
        }
        .blacklisted-recommendation {
            border-left: 5px solid #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get broker intelligence data from service
        intel_data = data_service.get_broker_intelligence_data(account_type=account_type)
        
        # If no intelligence data, show placeholder
        if not intel_data:
            st.info("Broker Intelligence system is not active or no data available.")
            return
        
        # Health Status Display
        _render_health_status(intel_data.get("health_status", "NORMAL"))
        
        # Create tabs for different views
        if not simplified:
            tabs = st.tabs(["Broker Health", "Circuit Breakers", "Recommendations", "Alert Log"])
            
            with tabs[0]:
                _render_broker_health_tab(intel_data, data_service, account_type)
            
            with tabs[1]:
                _render_circuit_breakers_tab(intel_data)
            
            with tabs[2]:
                _render_recommendations_tab(intel_data, data_service, account_type)
            
            with tabs[3]:
                _render_alerts_tab(intel_data)
        else:
            # Simplified view shows just key metrics
            _render_broker_health_simplified(intel_data)
        
    except Exception as e:
        logger.error(f"Error rendering broker intelligence: {str(e)}")
        st.error(f"Error rendering broker intelligence: {str(e)}")

def _render_health_status(status: str):
    """Render health status badge"""
    if status == "NORMAL":
        status_class = "status-badge status-normal"
        status_text = "NORMAL"
    elif status == "CAUTION":
        status_class = "status-badge status-caution"
        status_text = "CAUTION"
    else:  # CRITICAL
        status_class = "status-badge status-critical"
        status_text = "CRITICAL"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h4 style="margin-right: 10px; margin-bottom: 0;">System Health Status:</h4>
        <div class="{status_class}">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

def _render_broker_health_tab(intel_data: Dict[str, Any], data_service: DataService, account_type: str):
    """Render broker health tab"""
    # Get list of registered brokers
    brokers = data_service.get_registered_brokers(account_type=account_type)
    
    if not brokers:
        st.info("No brokers registered")
        return
    
    # Broker selector
    selected_broker = st.selectbox(
        "Select Broker:",
        options=brokers,
        key="intelligence_broker_selector"
    )
    
    if not selected_broker:
        st.info("Select a broker to view health metrics")
        return
    
    # Get health report for the selected broker
    health_report = intel_data.get("health_report", {})
    broker_data = health_report.get("brokers", {}).get(selected_broker, {})
    
    if not broker_data:
        st.info(f"No health data available for {selected_broker}")
        return
    
    # Extract metrics
    performance_score = broker_data.get("performance_score", 0)
    factor_scores = broker_data.get("factor_scores", {})
    circuit_breaker_active = broker_data.get("circuit_breaker_active", False)
    
    # Performance Score Cards
    score_color = _get_score_color(performance_score)
    
    cols = st.columns(5)
    
    # Overall score
    with cols[0]:
        st.metric(
            "Overall Score", 
            f"{performance_score:.1f}",
            delta=None,
            delta_color="off"
        )
        st.markdown(f"""
        <div style="text-align: center;">
            <span>Circuit Breaker: </span>
            <span style="color: {'#dc3545' if circuit_breaker_active else '#28a745'}; font-weight: bold;">
                {'ACTIVE' if circuit_breaker_active else 'Inactive'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Factor scores
    factor_idx = 1
    for factor, score in factor_scores.items():
        if factor == "circuit_breaker" or factor_idx >= 5:
            continue
        
        factor_name = factor.replace("_", " ").title()
        
        with cols[factor_idx]:
            st.metric(
                factor_name,
                f"{score:.1f}",
                delta=None,
                delta_color="off"
            )
        
        factor_idx += 1
    
    # Performance breakdown chart
    st.subheader("Performance Breakdown")
    
    # Remove circuit breaker from factors (it's binary)
    factor_scores_chart = {k: v for k, v in factor_scores.items() if k != "circuit_breaker"}
    
    # Create dataframe for visualization
    df = pd.DataFrame({
        "Factor": [f.replace("_", " ").title() for f in factor_scores_chart.keys()],
        "Score": list(factor_scores_chart.values())
    })
    
    # Sort by score (ascending)
    df = df.sort_values("Score")
    
    # Create bar chart
    fig = px.bar(
        df,
        x="Score",
        y="Factor",
        orientation="h",
        color="Score",
        color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
        range_color=[0, 100]
    )
    
    # Update layout
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=20, b=0),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical performance
    st.subheader("Historical Performance")
    
    # In a real implementation, this would fetch historical data
    # from a database. For now, we'll generate mock data for visualization.
    
    # Generate mock timestamps (last 24 hours, hourly)
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    
    # Generate mock scores with some realistic variation
    base_score = 85  # Start with a good score
    np.random.seed(hash(selected_broker) % 1000)  # Consistent randomness for demo
    variations = np.random.normal(0, 5, len(timestamps))  # Random variations
    
    # Apply some systematic changes for realism
    for i in range(len(variations)):
        # Add a dip around 8 hours ago
        if 7 <= i <= 9:
            variations[i] -= 15
    
    scores = [max(0, min(100, base_score + v)) for v in variations]
    
    # Create dataframe
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Overall Score": scores
    })
    
    # Create line chart
    fig = px.line(
        df,
        x="Timestamp",
        y="Overall Score",
        markers=True
    )
    
    # Add threshold reference lines
    fig.add_shape(
        type="line",
        x0=df["Timestamp"].min(),
        y0=70,
        x1=df["Timestamp"].max(),
        y1=70,
        line=dict(color="#ffc107", width=2, dash="dash"),
        name="Caution Threshold"
    )
    
    fig.add_shape(
        type="line",
        x0=df["Timestamp"].min(),
        y0=40,
        x1=df["Timestamp"].max(),
        y1=40,
        line=dict(color="#dc3545", width=2, dash="dash"),
        name="Critical Threshold"
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _render_broker_health_simplified(intel_data: Dict[str, Any]):
    """Render simplified broker health view"""
    health_report = intel_data.get("health_report", {})
    brokers_data = health_report.get("brokers", {})
    
    if not brokers_data:
        st.info("No broker health data available")
        return
    
    # Create columns for broker health cards
    num_brokers = len(brokers_data)
    cols_per_row = min(3, num_brokers)
    
    # Create broker score cards
    for i in range(0, num_brokers, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < num_brokers:
                broker_id = list(brokers_data.keys())[idx]
                broker_data = brokers_data[broker_id]
                
                performance_score = broker_data.get("performance_score", 0)
                circuit_breaker_active = broker_data.get("circuit_breaker_active", False)
                
                with cols[j]:
                    st.metric(
                        broker_id,
                        f"{performance_score:.1f}",
                        delta=None,
                        delta_color="off"
                    )
                    
                    if circuit_breaker_active:
                        st.markdown("""
                        <div style="text-align: center; color: #dc3545; font-weight: bold;">
                            ⚠️ Circuit Breaker Active
                        </div>
                        """, unsafe_allow_html=True)

def _render_circuit_breakers_tab(intel_data: Dict[str, Any]):
    """Render circuit breakers tab"""
    # Get active circuit breakers
    circuit_breakers = intel_data.get("circuit_breakers", {})
    
    if not circuit_breakers:
        st.success("No Active Circuit Breakers - All brokers are operating normally")
        return
    
    # Display each active circuit breaker
    for broker_id, state in circuit_breakers.items():
        if not state.get("active", False):
            continue
        
        # Get details
        reason = state.get("reason", "Unknown")
        tripped_at = state.get("tripped_at", 0)
        reset_time = state.get("reset_time", 0)
        
        # Calculate time remaining until auto-reset
        now = time.time()
        seconds_remaining = max(0, reset_time - now)
        minutes_remaining = int(seconds_remaining / 60)
        
        # Format trip time
        trip_time = datetime.fromtimestamp(tripped_at).strftime("%Y-%m-%d %H:%M:%S")
        
        # Display alert
        st.markdown(f"""
        <div class="circuit-breaker-alert">
            <h5 style="color: #dc3545;">Circuit Breaker Active: {broker_id}</h5>
            <p><strong>Reason:</strong> {reason}</p>
            <p><strong>Tripped At:</strong> {trip_time}</p>
            <p><strong>Auto-Reset In:</strong> {"" if minutes_remaining > 0 else "Imminent"}{f"{minutes_remaining} minutes" if minutes_remaining > 0 else ""}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress = max(0, min(100, (seconds_remaining / state.get("reset_after", 300)) * 100))
        st.progress(progress / 100)

def _render_recommendations_tab(intel_data: Dict[str, Any], data_service: DataService, account_type: str):
    """Render recommendations tab"""
    # Asset class and operation type selectors
    col1, col2 = st.columns(2)
    
    with col1:
        asset_class = st.selectbox(
            "Asset Class:",
            options=["equities", "forex", "futures", "options", "crypto"],
            index=0,
            key="asset_class_selector"
        )
    
    with col2:
        operation_type = st.selectbox(
            "Operation Type:",
            options=["order", "data", "quote"],
            index=0,
            key="operation_type_selector"
        )
    
    # Get recommendation
    recommendations = intel_data.get("recommendations", {})
    advice_key = f"{asset_class}_{operation_type}"
    advice = recommendations.get(advice_key, {})
    
    if not advice:
        st.info(f"No recommendation available for {asset_class}/{operation_type}")
        return
    
    # Display recommendation
    failover_recommended = advice.get("is_failover_recommended", False)
    primary_broker = advice.get("primary_broker_id")
    backup_brokers = advice.get("backup_broker_ids", [])
    blacklisted_brokers = advice.get("blacklisted_broker_ids", [])
    priority_scores = advice.get("priority_scores", {})
    advisory_notes = advice.get("advisory_notes", [])
    
    # Status banner
    status_text = "FAILOVER RECOMMENDED" if failover_recommended else "NORMAL"
    status_color = "#dc3545" if failover_recommended else "#28a745"
    
    st.markdown(f"""
    <h5>
        Recommendation Status: 
        <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
    </h5>
    """, unsafe_allow_html=True)
    
    # Advisory notes
    if advisory_notes:
        st.subheader("Advisory Notes")
        for note in advisory_notes:
            st.markdown(f"- {note}")
    
    # Primary recommendation
    if primary_broker:
        primary_score = priority_scores.get(primary_broker, 0)
        
        st.subheader("Primary Recommendation")
        st.markdown(f"""
        <div class="recommendation-card primary-recommendation">
            <h4>{primary_broker}</h4>
            <div><strong>Performance Score:</strong> {primary_score:.1f}</div>
            <div><strong>Asset Class:</strong> {asset_class}</div>
            <div><strong>Operation:</strong> {operation_type}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Backup recommendations
    if backup_brokers:
        st.subheader("Backup Options")
        
        # Create columns for backup brokers
        cols_per_row = min(3, len(backup_brokers))
        cols = st.columns(cols_per_row)
        
        for i, broker_id in enumerate(backup_brokers):
            broker_score = priority_scores.get(broker_id, 0)
            
            with cols[i % cols_per_row]:
                st.markdown(f"""
                <div class="recommendation-card backup-recommendation">
                    <h5>{broker_id}</h5>
                    <div><strong>Score:</strong> {broker_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Blacklisted brokers
    if blacklisted_brokers:
        st.subheader("Blacklisted Brokers")
        st.markdown('<div style="color: #dc3545;">', unsafe_allow_html=True)
        for broker_id in blacklisted_brokers:
            st.markdown(f"- **{broker_id}** - Circuit breaker active")
        st.markdown('</div>', unsafe_allow_html=True)

def _render_alerts_tab(intel_data: Dict[str, Any]):
    """Render alerts tab"""
    # In a production system, you would fetch actual alerts from a database
    # For demonstration, we'll create mock alerts
    
    # Mock alerts (in a real system, these would come from a database)
    mock_alerts = intel_data.get("alerts", [
        {
            "timestamp": datetime.now() - timedelta(minutes=5),
            "type": "circuit_breaker",
            "level": "high",
            "message": "Circuit breaker tripped for broker 'tradier' due to high error rate",
            "details": "Error rate exceeded 30% threshold"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=15),
            "type": "health_status",
            "level": "medium",
            "message": "System health status changed to CAUTION",
            "details": "Multiple brokers experiencing elevated latency"
        },
        {
            "timestamp": datetime.now() - timedelta(hours=1),
            "type": "recommendation",
            "level": "medium",
            "message": "Failover recommended for equities/order operations",
            "details": "Primary: alpaca, Current: tradier (20.5% better performance)"
        },
        {
            "timestamp": datetime.now() - timedelta(hours=2),
            "type": "reliability",
            "level": "low",
            "message": "Reliability metrics for broker 'interactive_brokers' improving",
            "details": "Error rate decreased from 15% to 2%"
        }
    ])
    
    if not mock_alerts:
        st.success("No Intelligence Alerts - System is operating normally")
        return
    
    # Display alerts
    for alert in mock_alerts:
        # Format timestamp
        timestamp = alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine alert color based on level
        if alert["level"] == "high":
            color = "#dc3545"  # Red
            border_color = "#dc3545"
        elif alert["level"] == "medium":
            color = "#ffc107"  # Yellow
            border_color = "#ffc107"
        else:  # low
            color = "#17a2b8"  # Info blue
            border_color = "#17a2b8"
        
        st.markdown(f"""
        <div style="padding: 10px 15px; margin-bottom: 10px; border-left: 5px solid {border_color}; background-color: rgba({int(border_color[1:3], 16)}, {int(border_color[3:5], 16)}, {int(border_color[5:7], 16)}, 0.1);">
            <div>
                <strong>{alert["message"]}</strong>
                <span style="color: #6c757d; margin-left: 10px;">({timestamp})</span>
            </div>
            <p style="margin-bottom: 0; margin-top: 5px;">{alert["details"]}</p>
        </div>
        """, unsafe_allow_html=True)

def _get_score_color(score: float) -> str:
    """Get color for a score based on its value"""
    if score >= 80:
        return "#28a745"  # Green (good)
    elif score >= 50:
        return "#ffc107"  # Yellow (caution)
    else:
        return "#dc3545"  # Red (poor)
