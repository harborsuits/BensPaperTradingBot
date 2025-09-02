#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Trading System Dashboard

This module implements the main dashboard application for the enhanced trading system,
integrating cross-asset opportunity visualization, strategy performance feedback,
capital allocation insights, and risk management metrics.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

# Import trading system components
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.core.cross_asset_opportunity_ranker import CrossAssetOpportunityRanker
from trading_bot.core.capital_allocation_optimizer import CapitalAllocationOptimizer
from trading_bot.core.cross_asset_risk_manager import CrossAssetRiskManager
from trading_bot.core.strategy_performance_feedback import StrategyPerformanceFeedback

# Import dashboard components
from dashboard.components.cross_asset_dashboard import CrossAssetDashboard, custom_css
from dashboard.components.strategy_manager_dashboard import StrategyManagerDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dashboard")

# Initialize event bus
event_bus = EventBus()

# Initialize enhanced system components
opportunity_ranker = CrossAssetOpportunityRanker(event_bus)
allocation_optimizer = CapitalAllocationOptimizer(event_bus)
risk_manager = CrossAssetRiskManager(event_bus)
strategy_performance = StrategyPerformanceFeedback(event_bus)

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
    ],
    suppress_callback_exceptions=True
)

# Initialize enhanced strategy manager (will be connected later in main.py)
strategy_manager = None

# Create dashboard components
cross_asset_dashboard = CrossAssetDashboard(app, event_bus)
strategy_dashboard = StrategyManagerDashboard(app, event_bus, strategy_manager)

# App layout
app.layout = html.Div([
    # Custom CSS
    html.Style(custom_css),
    
    # Navigation bar
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H2("Enhanced Trading System", className="text-white mb-0")),
            ], className="flex-grow-1"),
            dbc.Row([
                dbc.Col(html.Div(id="current-time", className="text-white")),
            ])
        ]),
        color="dark",
        dark=True,
        className="mb-4"
    ),
    
    # Main content
    dbc.Container([
        # Market regime indicator
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Current Market Regime"),
                    dbc.CardBody([
                        html.H3(id="market-regime", className="text-center"),
                        html.Div(id="regime-indicators", className="d-flex justify-content-center")
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Main dashboard content
        dbc.Tabs([
            dbc.Tab(cross_asset_dashboard.layout(), label="Cross-Asset Opportunities", tab_id="tab-opportunities"),
            dbc.Tab(strategy_dashboard.layout(), label="Strategy Manager", tab_id="tab-strategies"),
            dbc.Tab(html.Div(id="tab-portfolio-content"), label="Portfolio Analysis", tab_id="tab-portfolio"),
            dbc.Tab(html.Div(id="tab-performance-content"), label="Performance Analysis", tab_id="tab-performance"),
            dbc.Tab(html.Div(id="tab-settings-content"), label="System Settings", tab_id="tab-settings")
        ], id="main-tabs"),
        
        # Update interval for real-time data
        dcc.Interval(
            id="update-interval",
            interval=30 * 1000,  # 30 seconds
            n_intervals=0
        ),
        
        # Store for current market regime
        dcc.Store(id="market-regime-store")
    ])
])

# Callback to update current time
@app.callback(
    Output("current-time", "children"),
    [Input("update-interval", "n_intervals")]
)
def update_time(n_intervals):
    """Update current time display."""
    return f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Callback to update market regime display
@app.callback(
    [Output("market-regime", "children"),
     Output("market-regime", "className"),
     Output("regime-indicators", "children")],
    [Input("market-regime-store", "data")]
)
def update_market_regime_display(regime_data):
    """Update market regime display."""
    if not regime_data:
        regime = "Unknown"
        color_class = "text-secondary"
        indicators = []
    else:
        regime = regime_data.get("regime", "Unknown").capitalize()
        
        # Determine color based on regime
        if regime.lower() == "bullish":
            color_class = "text-success"
            icon = "arrow-up"
        elif regime.lower() == "bearish":
            color_class = "text-danger"
            icon = "arrow-down"
        elif regime.lower() == "volatile":
            color_class = "text-warning"
            icon = "bolt"
        elif regime.lower() == "sideways":
            color_class = "text-info"
            icon = "arrows-alt-h"
        else:
            color_class = "text-secondary"
            icon = "question-circle"
        
        # Create indicators
        vix = regime_data.get("vix", 0)
        trend_strength = regime_data.get("trend_strength", 0)
        
        indicators = [
            dbc.Badge(f"VIX: {vix:.2f}", color="light", className="mx-1"),
            dbc.Badge(f"Trend Strength: {trend_strength:.2f}", color="light", className="mx-1")
        ]
    
    return regime, f"text-center {color_class}", indicators

# Callback to handle tab selection
@app.callback(
    [Output("tab-portfolio-content", "children"),
     Output("tab-performance-content", "children"),
     Output("tab-settings-content", "children")],
    [Input("main-tabs", "active_tab")]
)
def render_tab_content(active_tab):
    """Render content for selected tab."""
    # Portfolio Analysis tab
    if active_tab == "tab-portfolio":
        portfolio_content = html.Div([
            html.H3("Portfolio Analysis"),
            html.P("Portfolio analysis content will be shown here."),
            # Future portfolio analysis components will be integrated here
        ])
    else:
        portfolio_content = html.Div()
    
    # Performance Analysis tab
    if active_tab == "tab-performance":
        performance_content = html.Div([
            html.H3("Performance Analysis"),
            html.P("Performance analysis content will be shown here."),
            # Future performance analysis components will be integrated here
        ])
    else:
        performance_content = html.Div()
    
    # System Settings tab
    if active_tab == "tab-settings":
        settings_content = html.Div([
            html.H3("System Settings"),
            dbc.Card([
                dbc.CardHeader("Enhanced Components"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Cross-Asset Opportunity Ranker"),
                                dbc.Switch(
                                    id="switch-opportunity-ranker", 
                                    value=True, 
                                    label="Enabled"
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Capital Allocation Optimizer"),
                                dbc.Switch(
                                    id="switch-allocation-optimizer", 
                                    value=True, 
                                    label="Enabled"
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Cross-Asset Risk Manager"),
                                dbc.Switch(
                                    id="switch-risk-manager", 
                                    value=True, 
                                    label="Enabled"
                                )
                            ], md=4)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Strategy Performance Feedback"),
                                dbc.Switch(
                                    id="switch-performance-feedback", 
                                    value=True, 
                                    label="Enabled"
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Signal Quality Enhancer"),
                                dbc.Switch(
                                    id="switch-signal-enhancer", 
                                    value=True, 
                                    label="Enabled"
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Data Flow Enhancer"),
                                dbc.Switch(
                                    id="switch-data-flow", 
                                    value=True, 
                                    label="Enabled"
                                )
                            ], md=4)
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Risk Management Parameters"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Max Single Asset Exposure"),
                                dbc.Input(
                                    id="input-max-asset-exposure",
                                    type="number",
                                    min=0.01,
                                    max=0.5,
                                    step=0.01,
                                    value=0.15,
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Max Asset Class Exposure"),
                                dbc.Input(
                                    id="input-max-class-exposure",
                                    type="number",
                                    min=0.1,
                                    max=0.8,
                                    step=0.05,
                                    value=0.4,
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Correlation Threshold"),
                                dbc.Input(
                                    id="input-correlation-threshold",
                                    type="number",
                                    min=0.3,
                                    max=0.9,
                                    step=0.05,
                                    value=0.7,
                                )
                            ], md=4)
                        ])
                    ])
                ])
            ])
        ])
    else:
        settings_content = html.Div()
    
    return portfolio_content, performance_content, settings_content

# Run server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
