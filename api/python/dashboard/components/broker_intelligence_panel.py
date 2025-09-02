#!/usr/bin/env python3
"""
Broker Intelligence Dashboard Panel

Provides real-time visualization of broker intelligence metrics and recommendations
for the BensBot trading dashboard, including health status, performance scores,
circuit breaker alerts, and failover recommendations.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

from trading_bot.brokers.metrics.base import MetricType, MetricOperation, MetricPeriod
from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.intelligence.broker_advisor import BrokerSelectionFactor
from trading_bot.brokers.intelligence.multi_broker_integration import BrokerIntelligenceEngine

# Configure logging
logger = logging.getLogger(__name__)

class BrokerIntelligencePanel:
    """
    Dashboard panel for broker intelligence visualization
    
    Provides real-time visualization of broker health status, performance scores,
    circuit breaker alerts, and failover recommendations from the broker
    intelligence system.
    """
    
    def __init__(
        self,
        app: dash.Dash,
        intelligence_engine: BrokerIntelligenceEngine,
        update_interval: int = 15000  # 15 seconds in ms
    ):
        """
        Initialize broker intelligence panel
        
        Args:
            app: Dash application instance
            intelligence_engine: Broker intelligence engine
            update_interval: Update interval in milliseconds
        """
        self.app = app
        self.intelligence_engine = intelligence_engine
        self.update_interval = update_interval
        
        # Initialize panel
        self._init_layout()
        self._init_callbacks()
        
        logger.info("Initialized broker intelligence dashboard panel")
    
    def _init_layout(self):
        """Initialize panel layout"""
        # Main panel layout
        self.layout = html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Broker Intelligence Center", className="card-title"),
                    html.Span("Provides situational awareness without overriding orchestrator decisions", 
                             className="text-muted ml-2")
                ]),
                dbc.CardBody([
                    # System Health Status Banner
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("System Health Status:", className="d-inline mr-3"),
                                html.Div(id="broker-health-status-badge", className="d-inline")
                            ], className="health-status-banner mb-4 p-3")
                        ])
                    ]),
                    
                    # Main Tabs for different intelligence views
                    dbc.Tabs([
                        # Broker Health & Performance Tab
                        dbc.Tab([
                            dbc.Row([
                                # Broker selector 
                                dbc.Col([
                                    html.Label("Select Broker:"),
                                    dcc.Dropdown(
                                        id="intelligence-broker-selector",
                                        options=[],  # Will be populated in callback
                                        value=None,
                                        clearable=False
                                    )
                                ], width=4, className="mb-4")
                            ]),
                            
                            # Broker Health Cards
                            dbc.Row(id="broker-health-cards"),
                            
                            # Performance Score Breakdown
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Performance Score Breakdown", className="mt-4 mb-3"),
                                    dcc.Graph(id="broker-performance-breakdown")
                                ])
                            ]),
                            
                            # Historical Performance
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Historical Performance", className="mt-4 mb-3"),
                                    dcc.Graph(id="broker-historical-performance")
                                ])
                            ])
                        ], label="Broker Health", tab_id="tab-broker-health"),
                        
                        # Circuit Breakers Tab
                        dbc.Tab([
                            html.Div(id="circuit-breaker-alerts", className="mt-3")
                        ], label="Circuit Breakers", tab_id="tab-circuit-breakers"),
                        
                        # Broker Recommendations Tab
                        dbc.Tab([
                            # Asset Class & Operation Type Selectors
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Asset Class:"),
                                    dcc.Dropdown(
                                        id="asset-class-selector",
                                        options=[
                                            {"label": "Equities", "value": "equities"},
                                            {"label": "Forex", "value": "forex"},
                                            {"label": "Futures", "value": "futures"},
                                            {"label": "Options", "value": "options"},
                                            {"label": "Crypto", "value": "crypto"}
                                        ],
                                        value="equities",
                                        clearable=False
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Operation Type:"),
                                    dcc.Dropdown(
                                        id="operation-type-selector",
                                        options=[
                                            {"label": "Order Placement", "value": "order"},
                                            {"label": "Market Data", "value": "data"},
                                            {"label": "Quote Retrieval", "value": "quote"}
                                        ],
                                        value="order",
                                        clearable=False
                                    )
                                ], width=4)
                            ], className="mb-4"),
                            
                            # Recommendation Results
                            html.Div(id="broker-recommendation-results")
                        ], label="Recommendations", tab_id="tab-recommendations"),
                        
                        # Intelligence Alerts Tab
                        dbc.Tab([
                            html.Div(id="intelligence-alerts", className="mt-3")
                        ], label="Alert Log", tab_id="tab-alerts")
                    ], id="intelligence-tabs", active_tab="tab-broker-health"),
                    
                    # Automatic refresh timer
                    dcc.Interval(
                        id="intelligence-refresh-interval",
                        interval=self.update_interval,
                        n_intervals=0
                    )
                ])
            ])
        ])
        
        # Add custom CSS for the panel
        self.app.clientside_callback(
            """
            function(n_clicks) {
                document.querySelector('head').insertAdjacentHTML('beforeend', `
                    <style>
                        /* Health status banner styling */
                        .health-status-banner {
                            border-radius: 4px;
                            border: 1px solid #ddd;
                        }
                        
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
                        
                        /* Intelligence alert styling */
                        .intelligence-alert {
                            padding: 10px 15px;
                            margin-bottom: 10px;
                            border-radius: 4px;
                        }
                        .alert-high {
                            background-color: rgba(220, 53, 69, 0.1);
                            border-left: 5px solid #dc3545;
                        }
                        .alert-medium {
                            background-color: rgba(255, 193, 7, 0.1);
                            border-left: 5px solid #ffc107;
                        }
                        .alert-low {
                            background-color: rgba(23, 162, 184, 0.1);
                            border-left: 5px solid #17a2b8;
                        }
                    </style>
                `);
                return '';
            }
            """,
            Output("intelligence-refresh-interval", "disabled"),
            Input("intelligence-refresh-interval", "n_intervals"),
        )
    
    def _init_callbacks(self):
        """Initialize dashboard callbacks"""
        
        # Broker selector update callback
        @self.app.callback(
            Output("intelligence-broker-selector", "options"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_broker_selector(n_intervals):
            """Update broker selector options"""
            # Get all registered brokers
            registered_brokers = self.intelligence_engine.registered_brokers
            
            # Create options list
            options = [
                {"label": broker_id, "value": broker_id}
                for broker_id in registered_brokers.keys()
            ]
            
            # Also ensure a default selection
            if options and "intelligence-broker-selector" not in dash.callback_context.states:
                self.app.clientside_callback(
                    """
                    function(options) {
                        if (options && options.length > 0) {
                            return options[0].value;
                        }
                        return null;
                    }
                    """,
                    Output("intelligence-broker-selector", "value"),
                    Input("intelligence-broker-selector", "options")
                )
            
            return options
        
        # Health status badge update callback
        @self.app.callback(
            Output("broker-health-status-badge", "children"),
            Output("broker-health-status-badge", "className"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_health_status(n_intervals):
            """Update health status badge"""
            # Get current health status
            status = self.intelligence_engine.health_status
            
            # Determine status badge style
            if status == "NORMAL":
                status_class = "status-badge status-normal"
                status_text = "NORMAL"
            elif status == "CAUTION":
                status_class = "status-badge status-caution"
                status_text = "CAUTION"
            else:  # CRITICAL
                status_class = "status-badge status-critical"
                status_text = "CRITICAL"
            
            return status_text, status_class
        
        # Broker health cards update callback
        @self.app.callback(
            Output("broker-health-cards", "children"),
            Input("intelligence-broker-selector", "value"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_broker_health_cards(broker_id, n_intervals):
            """Update broker health cards"""
            if not broker_id:
                return html.Div("Select a broker to view health metrics")
            
            try:
                # Get health report from the intelligence engine
                health_report = self.intelligence_engine._generate_health_report()
                broker_data = health_report["brokers"].get(broker_id, {})
                
                # Extract metrics
                performance_score = broker_data.get("performance_score", 0)
                factor_scores = broker_data.get("factor_scores", {})
                circuit_breaker_active = broker_data.get("circuit_breaker_active", False)
                
                # Create health cards
                cards = []
                
                # Overall score card
                score_color = self._get_score_color(performance_score)
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Overall Score", className="card-title"),
                                html.H2(f"{performance_score:.1f}", 
                                        style={"color": score_color},
                                        className="text-center"),
                                html.P("0-100 Scale", className="text-muted text-center"),
                                html.Div([
                                    html.Span("Circuit Breaker: "),
                                    html.Span(
                                        "ACTIVE" if circuit_breaker_active else "Inactive",
                                        className="ml-2",
                                        style={
                                            "color": "#dc3545" if circuit_breaker_active else "#28a745",
                                            "font-weight": "bold"
                                        }
                                    )
                                ], className="text-center mt-2")
                            ])
                        ], className="h-100")
                    ], width=3)
                )
                
                # Factor score cards
                for factor, score in factor_scores.items():
                    if factor == "circuit_breaker":
                        continue
                    
                    # Get factor display name and score color
                    factor_name = factor.replace("_", " ").title()
                    score_color = self._get_score_color(score)
                    
                    cards.append(
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5(factor_name, className="card-title"),
                                    html.H2(f"{score:.1f}", 
                                            style={"color": score_color},
                                            className="text-center"),
                                    html.P("0-100 Scale", className="text-muted text-center")
                                ])
                            ], className="h-100")
                        ], width=2)
                    )
                
                return dbc.Row(cards)
                
            except Exception as e:
                logger.error(f"Error updating broker health cards: {str(e)}")
                return html.Div(f"Error loading broker health data: {str(e)}")
        
        # Performance breakdown chart callback
        @self.app.callback(
            Output("broker-performance-breakdown", "figure"),
            Input("intelligence-broker-selector", "value"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_performance_breakdown(broker_id, n_intervals):
            """Update performance breakdown chart"""
            if not broker_id:
                return self._empty_chart("Select a broker to view performance breakdown")
            
            try:
                # Get health report from the intelligence engine
                health_report = self.intelligence_engine._generate_health_report()
                broker_data = health_report["brokers"].get(broker_id, {})
                
                # Extract factor scores
                factor_scores = broker_data.get("factor_scores", {})
                
                # Remove circuit breaker from factors (it's binary)
                if "circuit_breaker" in factor_scores:
                    del factor_scores["circuit_breaker"]
                
                # Create dataframe for visualization
                df = pd.DataFrame({
                    "Factor": [f.replace("_", " ").title() for f in factor_scores.keys()],
                    "Score": list(factor_scores.values())
                })
                
                # Sort by score (ascending)
                df = df.sort_values("Score")
                
                # Create bar chart
                fig = px.bar(
                    df,
                    x="Score",
                    y="Factor",
                    orientation="h",
                    title=f"Performance Breakdown for {broker_id}",
                    color="Score",
                    color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                    range_color=[0, 100]
                )
                
                # Update layout
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    coloraxis_showscale=False
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating performance breakdown: {str(e)}")
                return self._empty_chart(f"Error: {str(e)}")
        
        # Historical performance chart callback
        @self.app.callback(
            Output("broker-historical-performance", "figure"),
            Input("intelligence-broker-selector", "value"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_historical_performance(broker_id, n_intervals):
            """Update historical performance chart"""
            if not broker_id:
                return self._empty_chart("Select a broker to view historical performance")
            
            try:
                # In a real implementation, this would fetch historical data
                # from a database. For now, we'll generate mock data for visualization.
                
                # Generate mock timestamps (last 24 hours, hourly)
                now = datetime.now()
                timestamps = [now - timedelta(hours=i) for i in range(24, 0, -1)]
                
                # Generate mock scores with some realistic variation
                base_score = 85  # Start with a good score
                np.random.seed(hash(broker_id) % 1000)  # Consistent randomness for demo
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
                    title=f"Historical Performance Trend for {broker_id}",
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
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis=dict(range=[0, 100])
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating historical performance: {str(e)}")
                return self._empty_chart(f"Error: {str(e)}")
        
        # Circuit breaker alerts callback
        @self.app.callback(
            Output("circuit-breaker-alerts", "children"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_circuit_breaker_alerts(n_intervals):
            """Update circuit breaker alerts"""
            try:
                # Get all active circuit breakers
                active_breakers = {}
                
                for broker_id, state in self.intelligence_engine.advisor.circuit_breakers.items():
                    if state.get("active", False):
                        active_breakers[broker_id] = state
                
                if not active_breakers:
                    return html.Div([
                        html.H5("No Active Circuit Breakers", className="text-center text-success mt-3"),
                        html.P("All brokers are operating normally", className="text-center text-muted")
                    ])
                
                # Create alert cards for each active circuit breaker
                alerts = []
                
                for broker_id, state in active_breakers.items():
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
                    
                    alerts.append(
                        html.Div([
                            html.H5(f"Circuit Breaker Active: {broker_id}", className="text-danger"),
                            html.P([
                                html.Strong("Reason: "), reason
                            ]),
                            html.P([
                                html.Strong("Tripped At: "), trip_time
                            ]),
                            html.P([
                                html.Strong("Auto-Reset In: "), 
                                f"{minutes_remaining} minutes" if minutes_remaining > 0 else "Imminent"
                            ]),
                            dbc.Progress(
                                value=max(0, min(100, (seconds_remaining / state.get("reset_after", 300)) * 100)),
                                color="danger",
                                className="mt-2 mb-2"
                            )
                        ], className="circuit-breaker-alert")
                    )
                
                return html.Div(alerts)
                
            except Exception as e:
                logger.error(f"Error updating circuit breaker alerts: {str(e)}")
                return html.Div(f"Error loading circuit breaker data: {str(e)}")
        
        # Broker recommendation results callback
        @self.app.callback(
            Output("broker-recommendation-results", "children"),
            Input("asset-class-selector", "value"),
            Input("operation-type-selector", "value"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_broker_recommendations(asset_class, operation_type, n_intervals):
            """Update broker recommendations"""
            if not asset_class or not operation_type:
                return html.Div("Select asset class and operation type to view recommendations")
            
            try:
                # Get recommendation from advisor
                advice = self.intelligence_engine.advisor.get_selection_advice(
                    asset_class=asset_class,
                    operation_type=operation_type
                )
                
                # Create recommendation display
                result = []
                
                # Top status banner
                failover_status = "FAILOVER RECOMMENDED" if advice.is_failover_recommended else "NORMAL"
                status_color = "#dc3545" if advice.is_failover_recommended else "#28a745"
                
                result.append(
                    html.Div([
                        html.H5([
                            "Recommendation Status: ",
                            html.Span(failover_status, style={"color": status_color, "font-weight": "bold"})
                        ], className="mb-3")
                    ])
                )
                
                # Advisory notes
                if advice.advisory_notes:
                    notes_list = [html.Li(note) for note in advice.advisory_notes]
                    result.append(
                        html.Div([
                            html.H6("Advisory Notes:"),
                            html.Ul(notes_list, className="mb-4")
                        ])
                    )
                
                # Primary recommendation
                if advice.primary_broker_id:
                    primary_score = advice.priority_scores.get(advice.primary_broker_id, 0)
                    result.append(
                        html.Div([
                            html.H6("Primary Recommendation:"),
                            html.Div([
                                html.H4(advice.primary_broker_id, className="mb-2"),
                                html.Div([
                                    html.Strong("Performance Score: "),
                                    html.Span(f"{primary_score:.1f}")
                                ]),
                                html.Div([
                                    html.Strong("Asset Class: "),
                                    html.Span(asset_class)
                                ]),
                                html.Div([
                                    html.Strong("Operation: "),
                                    html.Span(operation_type)
                                ])
                            ], className="recommendation-card primary-recommendation")
                        ])
                    )
                
                # Backup recommendations
                if advice.backup_broker_ids:
                    backup_cards = []
                    
                    for broker_id in advice.backup_broker_ids:
                        broker_score = advice.priority_scores.get(broker_id, 0)
                        backup_cards.append(
                            dbc.Col([
                                html.Div([
                                    html.H5(broker_id, className="mb-2"),
                                    html.Div([
                                        html.Strong("Score: "),
                                        html.Span(f"{broker_score:.1f}")
                                    ])
                                ], className="recommendation-card backup-recommendation h-100")
                            ], width=4)
                        )
                    
                    result.append(
                        html.Div([
                            html.H6("Backup Options:"),
                            dbc.Row(backup_cards)
                        ], className="mt-4")
                    )
                
                # Blacklisted brokers
                if advice.blacklisted_broker_ids:
                    blacklist_items = []
                    
                    for broker_id in advice.blacklisted_broker_ids:
                        blacklist_items.append(
                            html.Li([
                                html.Strong(broker_id),
                                " - Circuit breaker active"
                            ])
                        )
                    
                    result.append(
                        html.Div([
                            html.H6("Blacklisted Brokers:"),
                            html.Ul(blacklist_items, className="text-danger")
                        ], className="mt-4")
                    )
                
                return html.Div(result)
                
            except Exception as e:
                logger.error(f"Error updating broker recommendations: {str(e)}")
                return html.Div(f"Error loading recommendation data: {str(e)}")
        
        # Intelligence alerts callback
        @self.app.callback(
            Output("intelligence-alerts", "children"),
            Input("intelligence-refresh-interval", "n_intervals")
        )
        def update_intelligence_alerts(n_intervals):
            """Update intelligence alerts"""
            try:
                # In a production system, you would fetch actual alerts from a database
                # For demonstration, we'll create mock alerts
                
                # Mock alerts (in a real system, these would come from a database)
                mock_alerts = [
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
                ]
                
                if not mock_alerts:
                    return html.Div([
                        html.H5("No Intelligence Alerts", className="text-center text-success mt-3"),
                        html.P("System is operating normally", className="text-center text-muted")
                    ])
                
                # Create alert items
                alerts = []
                
                for alert in mock_alerts:
                    # Format timestamp
                    timestamp = alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Determine alert class based on level
                    alert_class = f"intelligence-alert alert-{alert['level']}"
                    
                    alerts.append(
                        html.Div([
                            html.Div([
                                html.Strong(alert["message"]),
                                html.Span(f" ({timestamp})", className="text-muted ml-2")
                            ]),
                            html.P(alert["details"], className="mb-0 mt-1")
                        ], className=alert_class)
                    )
                
                return html.Div(alerts)
                
            except Exception as e:
                logger.error(f"Error updating intelligence alerts: {str(e)}")
                return html.Div(f"Error loading alert data: {str(e)}")
    
    def _get_score_color(self, score: float) -> str:
        """Get color for a score based on its value"""
        if score >= 80:
            return "#28a745"  # Green (good)
        elif score >= 50:
            return "#ffc107"  # Yellow (caution)
        else:
            return "#dc3545"  # Red (poor)
    
    def _empty_chart(self, message: str = "No data available") -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    }
                }
            ],
            height=300
        )
        
        return fig
