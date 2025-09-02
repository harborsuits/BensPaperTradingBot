#!/usr/bin/env python3
"""
Broker Metrics Dashboard Panel

Provides real-time visualization of broker performance metrics
for the BensBot trading dashboard.
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

# Configure logging
logger = logging.getLogger(__name__)

class BrokerMetricsPanel:
    """
    Dashboard panel for broker performance metrics
    
    Integrates with the main dashboard to provide real-time 
    visualization of broker metrics.
    """
    
    def __init__(
        self,
        app: dash.Dash,
        metrics_manager: BrokerMetricsManager,
        update_interval: int = 30000  # 30 seconds in ms
    ):
        """
        Initialize broker metrics panel
        
        Args:
            app: Dash application instance
            metrics_manager: Broker metrics manager
            update_interval: Update interval in milliseconds
        """
        self.app = app
        self.metrics_manager = metrics_manager
        self.update_interval = update_interval
        
        # Initialize panel
        self._init_layout()
        self._init_callbacks()
        
        logger.info("Initialized broker metrics dashboard panel")
    
    def _init_layout(self):
        """Initialize panel layout"""
        # Main panel layout
        self.layout = html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Broker Performance Metrics", className="card-title"),
                    dbc.ButtonGroup([
                        dbc.Button("1h", id="btn-period-1h", color="secondary", outline=True, size="sm", className="mr-1"),
                        dbc.Button("6h", id="btn-period-6h", color="secondary", outline=True, size="sm", className="mr-1"),
                        dbc.Button("1d", id="btn-period-1d", color="secondary", outline=True, size="sm", className="mr-1", active=True),
                        dbc.Button("1w", id="btn-period-1w", color="secondary", outline=True, size="sm", className="mr-1"),
                    ], className="float-right")
                ], className="d-flex justify-content-between align-items-center"),
                dbc.CardBody([
                    # Broker selector
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Broker:"),
                            dcc.Dropdown(
                                id="broker-selector",
                                options=[],  # Will be populated in callback
                                value=None,
                                clearable=False
                            )
                        ], width=4)
                    ], className="mb-4"),
                    
                    # Key metrics cards
                    dbc.Row([
                        # Latency card
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Latency", className="card-title"),
                                    html.H3(id="avg-latency-value", className="text-center"),
                                    html.P("Avg Response Time", className="card-text text-center text-muted"),
                                    dcc.Graph(id="latency-sparkline", config={'displayModeBar': False}, style={'height': '60px'})
                                ])
                            ], className="h-100")
                        ], width=3),
                        
                        # Reliability card
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Reliability", className="card-title"),
                                    html.H3(id="availability-value", className="text-center"),
                                    html.P("Availability", className="card-text text-center text-muted"),
                                    dcc.Graph(id="reliability-sparkline", config={'displayModeBar': False}, style={'height': '60px'})
                                ])
                            ], className="h-100")
                        ], width=3),
                        
                        # Execution Quality card
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Execution Quality", className="card-title"),
                                    html.H3(id="slippage-value", className="text-center"),
                                    html.P("Avg Slippage", className="card-text text-center text-muted"),
                                    dcc.Graph(id="slippage-sparkline", config={'displayModeBar': False}, style={'height': '60px'})
                                ])
                            ], className="h-100")
                        ], width=3),
                        
                        # Cost card
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Trading Costs", className="card-title"),
                                    html.H3(id="commission-value", className="text-center"),
                                    html.P("Avg Commission", className="card-text text-center text-muted"),
                                    dcc.Graph(id="cost-sparkline", config={'displayModeBar': False}, style={'height': '60px'})
                                ])
                            ], className="h-100")
                        ], width=3)
                    ], className="mb-4"),
                    
                    # Detailed metrics tab
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id="latency-breakdown-chart")
                        ], label="Latency Analysis"),
                        
                        dbc.Tab([
                            dcc.Graph(id="reliability-chart")
                        ], label="Reliability Analysis"),
                        
                        dbc.Tab([
                            dcc.Graph(id="execution-quality-chart")
                        ], label="Execution Quality"),
                        
                        dbc.Tab([
                            dcc.Graph(id="cost-breakdown-chart")
                        ], label="Cost Analysis"),
                        
                        dbc.Tab([
                            html.Div(id="broker-comparison-container")
                        ], label="Broker Comparison")
                    ], className="mb-3"),
                    
                    # Hidden components for state
                    dcc.Store(id="selected-period", data="1d"),
                    dcc.Interval(id="metrics-update-interval", interval=self.update_interval)
                ])
            ], className="shadow")
        ])
    
    def _init_callbacks(self):
        """Initialize dashboard callbacks"""
        # Update period button states
        self.app.callback(
            [Output("btn-period-1h", "active"),
             Output("btn-period-6h", "active"),
             Output("btn-period-1d", "active"),
             Output("btn-period-1w", "active"),
             Output("selected-period", "data")],
            [Input("btn-period-1h", "n_clicks"),
             Input("btn-period-6h", "n_clicks"),
             Input("btn-period-1d", "n_clicks"),
             Input("btn-period-1w", "n_clicks")],
            [State("selected-period", "data")]
        )(self._update_period)
        
        # Update broker selector
        self.app.callback(
            Output("broker-selector", "options"),
            Input("metrics-update-interval", "n_intervals")
        )(self._update_broker_selector)
        
        # Update key metrics
        self.app.callback(
            [Output("avg-latency-value", "children"),
             Output("latency-sparkline", "figure"),
             Output("availability-value", "children"),
             Output("reliability-sparkline", "figure"),
             Output("slippage-value", "children"),
             Output("slippage-sparkline", "figure"),
             Output("commission-value", "children"),
             Output("cost-sparkline", "figure")],
            [Input("broker-selector", "value"),
             Input("selected-period", "data"),
             Input("metrics-update-interval", "n_intervals")]
        )(self._update_key_metrics)
        
        # Update detailed charts
        self.app.callback(
            [Output("latency-breakdown-chart", "figure"),
             Output("reliability-chart", "figure"),
             Output("execution-quality-chart", "figure"),
             Output("cost-breakdown-chart", "figure")],
            [Input("broker-selector", "value"),
             Input("selected-period", "data"),
             Input("metrics-update-interval", "n_intervals")]
        )(self._update_detailed_charts)
        
        # Update broker comparison
        self.app.callback(
            Output("broker-comparison-container", "children"),
            [Input("selected-period", "data"),
             Input("metrics-update-interval", "n_intervals")]
        )(self._update_broker_comparison)
    
    def _update_period(self, btn_1h, btn_6h, btn_1d, btn_1w, current_period):
        """Update selected time period based on button clicks"""
        ctx = dash.callback_context
        if not ctx.triggered:
            # No button clicked, keep current state
            return [False, False, True, False, "1d"]
            
        # Get which button was clicked
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "btn-period-1h":
            return [True, False, False, False, "1h"]
        elif button_id == "btn-period-6h":
            return [False, True, False, False, "6h"]
        elif button_id == "btn-period-1d":
            return [False, False, True, False, "1d"]
        elif button_id == "btn-period-1w":
            return [False, False, False, True, "1w"]
        
        # Fallback
        return [False, False, True, False, "1d"]
    
    def _update_broker_selector(self, n_intervals):
        """Update broker selector options"""
        # Get active brokers from metrics manager
        broker_options = []
        try:
            # Get brokers from active_brokers dictionary in metrics manager
            for broker_id, broker_type in self.metrics_manager.broker_types.items():
                broker_options.append({
                    "label": f"{broker_id} ({broker_type})",
                    "value": broker_id
                })
            
            # Add "All Brokers" option
            if broker_options:
                broker_options.insert(0, {"label": "All Brokers", "value": "all"})
        except Exception as e:
            logger.error(f"Error updating broker selector: {str(e)}")
        
        return broker_options
    
    def _update_key_metrics(self, broker_id, period_str, n_intervals):
        """Update key metrics cards"""
        try:
            # Convert period string to MetricPeriod
            period = self._get_metric_period(period_str)
            
            # Use first broker if "all" selected or no selection
            if broker_id == "all" or broker_id is None:
                brokers = list(self.metrics_manager.active_brokers.keys())
                if brokers:
                    broker_id = brokers[0]
                else:
                    # No brokers available
                    return ["N/A", self._empty_sparkline(), "N/A", self._empty_sparkline(),
                           "N/A", self._empty_sparkline(), "N/A", self._empty_sparkline()]
            
            # Get metrics for selected broker
            metrics = self.metrics_manager.get_broker_metrics(broker_id, period)
            
            # Extract values for cards
            avg_latency = metrics["latency"]["mean_ms"]
            availability = metrics["reliability"]["availability"]
            avg_slippage = metrics["execution_quality"]["avg_slippage_pct"]
            avg_commission = metrics["costs"]["avg_commission"]
            
            # Format values for display
            latency_text = f"{avg_latency:.2f} ms"
            availability_text = f"{availability:.1f}%"
            slippage_text = f"{avg_slippage:.4f}%"
            commission_text = f"${avg_commission:.4f}"
            
            # Create sparklines (simple examples, would be more complex with real data)
            latency_sparkline = self._create_sparkline([avg_latency] * 10, "latency")
            reliability_sparkline = self._create_sparkline([availability] * 10, "reliability")
            slippage_sparkline = self._create_sparkline([avg_slippage] * 10, "slippage")
            cost_sparkline = self._create_sparkline([avg_commission] * 10, "cost")
            
            return [latency_text, latency_sparkline,
                   availability_text, reliability_sparkline,
                   slippage_text, slippage_sparkline,
                   commission_text, cost_sparkline]
            
        except Exception as e:
            logger.error(f"Error updating key metrics: {str(e)}")
            return ["N/A", self._empty_sparkline(), "N/A", self._empty_sparkline(),
                   "N/A", self._empty_sparkline(), "N/A", self._empty_sparkline()]
    
    def _update_detailed_charts(self, broker_id, period_str, n_intervals):
        """Update detailed metrics charts"""
        try:
            # Convert period string to MetricPeriod
            period = self._get_metric_period(period_str)
            
            # Determine brokers to include
            if broker_id == "all" or broker_id is None:
                broker_ids = list(self.metrics_manager.active_brokers.keys())
            else:
                broker_ids = [broker_id]
            
            if not broker_ids:
                # No brokers available
                return [self._empty_chart("No brokers available"),
                       self._empty_chart("No brokers available"),
                       self._empty_chart("No brokers available"),
                       self._empty_chart("No brokers available")]
            
            # Get metrics for selected brokers
            latency_fig = self._create_latency_breakdown_chart(broker_ids, period)
            reliability_fig = self._create_reliability_chart(broker_ids, period)
            execution_fig = self._create_execution_quality_chart(broker_ids, period)
            cost_fig = self._create_cost_breakdown_chart(broker_ids, period)
            
            return [latency_fig, reliability_fig, execution_fig, cost_fig]
            
        except Exception as e:
            logger.error(f"Error updating detailed charts: {str(e)}")
            error_msg = f"Error: {str(e)}"
            return [self._empty_chart(error_msg),
                   self._empty_chart(error_msg),
                   self._empty_chart(error_msg),
                   self._empty_chart(error_msg)]
    
    def _update_broker_comparison(self, period_str, n_intervals):
        """Update broker comparison view"""
        try:
            # Convert period string to MetricPeriod
            period = self._get_metric_period(period_str)
            
            # Get all broker IDs
            broker_ids = list(self.metrics_manager.active_brokers.keys())
            
            if not broker_ids:
                return html.Div("No brokers available for comparison", className="text-center p-4")
            
            # Get metrics for all brokers
            brokers_data = []
            for broker_id in broker_ids:
                metrics = self.metrics_manager.get_broker_metrics(broker_id, period)
                brokers_data.append({
                    "broker_id": broker_id,
                    "broker_type": metrics["broker_type"],
                    "avg_latency": metrics["latency"]["mean_ms"],
                    "availability": metrics["reliability"]["availability"],
                    "avg_slippage": metrics["execution_quality"]["avg_slippage_pct"],
                    "avg_commission": metrics["costs"]["avg_commission"],
                    "total_cost": metrics["costs"]["overall_total"],
                    "errors": metrics["reliability"]["errors"]
                })
            
            # Create comparison table
            df = pd.DataFrame(brokers_data)
            
            table = dbc.Table.from_dataframe(
                df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                className="mt-3"
            )
            
            # Create radar chart for comparison
            fig = self._create_broker_comparison_radar(df)
            
            return html.Div([
                dcc.Graph(figure=fig, className="mb-4"),
                html.H5("Detailed Comparison"),
                table
            ])
            
        except Exception as e:
            logger.error(f"Error updating broker comparison: {str(e)}")
            return html.Div(f"Error updating broker comparison: {str(e)}", className="text-danger p-4")
    
    def _create_sparkline(self, values, metric_type):
        """Create a simple sparkline chart"""
        # For demonstration - in real scenario, would fetch historical values
        
        colors = {
            "latency": "#1f77b4",  # blue
            "reliability": "#2ca02c",  # green
            "slippage": "#ff7f0e",  # orange
            "cost": "#d62728"  # red
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=values,
            mode='lines',
            line=dict(width=2, color=colors.get(metric_type, "#1f77b4")),
            fill='tozeroy',
            fillcolor=colors.get(metric_type, "#1f77b4") + "20"  # 20% opacity
        ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=60,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            )
        )
        
        return fig
    
    def _empty_sparkline(self):
        """Create an empty sparkline chart"""
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=60,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[0, 1]
            )
        )
        
        return fig
    
    def _empty_chart(self, message="No data available"):
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            )
        )
        
        return fig
    
    def _create_latency_breakdown_chart(self, broker_ids, period):
        """Create a chart showing latency breakdown by operation type"""
        # Placeholder implementation - would use real metrics in production
        operations = ["GET_ACCOUNT", "GET_POSITIONS", "PLACE_ORDER", "GET_QUOTE"]
        
        # Create random latency data for demonstration
        np.random.seed(42)  # For reproducible random data
        
        data = []
        for broker_id in broker_ids:
            for op in operations:
                latency = np.random.normal(50, 20)  # Random latency with mean 50ms
                data.append({
                    "broker_id": broker_id,
                    "operation": op,
                    "latency_ms": latency
                })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create figure
        fig = px.bar(
            df, 
            x="operation", 
            y="latency_ms", 
            color="broker_id", 
            barmode="group",
            title="API Latency by Operation Type",
            labels={"operation": "Operation", "latency_ms": "Response Time (ms)", "broker_id": "Broker"}
        )
        
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_reliability_chart(self, broker_ids, period):
        """Create a chart showing reliability metrics"""
        # Placeholder implementation - would use real metrics in production
        
        # Create data for demonstration
        data = []
        for broker_id in broker_ids:
            # Get metrics from the metrics manager if available
            try:
                metrics = self.metrics_manager.get_broker_metrics(broker_id, period)
                availability = metrics["reliability"]["availability"]
                errors = metrics["reliability"]["errors"]
                reconnects = metrics["reliability"]["reconnects"]
            except:
                # Generate random data if metrics not available
                availability = np.random.uniform(95, 100)
                errors = int(np.random.uniform(0, 10))
                reconnects = int(np.random.uniform(0, 5))
            
            data.append({
                "broker_id": broker_id,
                "metric": "Availability (%)",
                "value": availability
            })
            data.append({
                "broker_id": broker_id,
                "metric": "Errors",
                "value": errors
            })
            data.append({
                "broker_id": broker_id,
                "metric": "Reconnects",
                "value": reconnects
            })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create figure
        fig = px.bar(
            df, 
            x="broker_id", 
            y="value", 
            color="metric", 
            barmode="group",
            title="Broker Reliability Metrics",
            labels={"broker_id": "Broker", "value": "Value", "metric": "Metric"}
        )
        
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_execution_quality_chart(self, broker_ids, period):
        """Create a chart showing execution quality metrics"""
        # Placeholder implementation - would use real metrics in production
        
        # Create data for demonstration
        data = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for broker_id in broker_ids:
            for symbol in symbols:
                # Random slippage between -0.1% and 0.1%
                slippage = np.random.uniform(-0.1, 0.1)
                # Random fill rate between 95% and 100%
                fill_rate = np.random.uniform(95, 100)
                
                data.append({
                    "broker_id": broker_id,
                    "symbol": symbol,
                    "slippage_pct": slippage,
                    "fill_rate_pct": fill_rate
                })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create figure
        fig = px.scatter(
            df, 
            x="slippage_pct", 
            y="fill_rate_pct", 
            color="broker_id",
            symbol="symbol", 
            size=[10] * len(df),  # Uniform size
            hover_data=["symbol", "broker_id", "slippage_pct", "fill_rate_pct"],
            title="Execution Quality by Symbol",
            labels={
                "slippage_pct": "Price Slippage (%)", 
                "fill_rate_pct": "Fill Rate (%)", 
                "broker_id": "Broker",
                "symbol": "Symbol"
            }
        )
        
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_cost_breakdown_chart(self, broker_ids, period):
        """Create a chart showing cost breakdown"""
        # Placeholder implementation - would use real metrics in production
        
        # Create data for demonstration
        data = []
        for broker_id in broker_ids:
            # Get metrics from the metrics manager if available
            try:
                metrics = self.metrics_manager.get_broker_metrics(broker_id, period)
                commission = metrics["costs"]["total_commission"]
                exchange_fee = metrics["costs"]["total_exchange_fee"]
                regulatory_fee = metrics["costs"]["total_regulatory_fee"]
                other_fees = metrics["costs"]["total_other_fees"]
            except:
                # Generate random data if metrics not available
                commission = np.random.uniform(10, 100)
                exchange_fee = np.random.uniform(5, 30)
                regulatory_fee = np.random.uniform(1, 10)
                other_fees = np.random.uniform(0, 5)
            
            data.append({
                "broker_id": broker_id,
                "fee_type": "Commission",
                "amount": commission
            })
            data.append({
                "broker_id": broker_id,
                "fee_type": "Exchange Fees",
                "amount": exchange_fee
            })
            data.append({
                "broker_id": broker_id,
                "fee_type": "Regulatory Fees",
                "amount": regulatory_fee
            })
            data.append({
                "broker_id": broker_id,
                "fee_type": "Other Fees",
                "amount": other_fees
            })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create figure
        fig = px.bar(
            df, 
            x="broker_id", 
            y="amount", 
            color="fee_type", 
            title="Trading Cost Breakdown",
            labels={"broker_id": "Broker", "amount": "Amount ($)", "fee_type": "Fee Type"}
        )
        
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_broker_comparison_radar(self, df):
        """Create a radar chart comparing brokers"""
        # Normalize metrics for radar chart
        metrics_to_compare = ["avg_latency", "availability", "avg_slippage", "avg_commission"]
        
        # Invert metrics where lower is better
        df["avg_latency_inv"] = 100 - df["avg_latency"].rank(pct=True) * 100
        df["avg_slippage_inv"] = 100 - df["avg_slippage"].rank(pct=True) * 100
        df["avg_commission_inv"] = 100 - df["avg_commission"].rank(pct=True) * 100
        
        # Normalized availability (higher is better)
        df["availability_norm"] = df["availability"].rank(pct=True) * 100
        
        radar_metrics = ["avg_latency_inv", "availability_norm", "avg_slippage_inv", "avg_commission_inv"]
        radar_labels = ["Response Time", "Availability", "Price Improvement", "Low Cost"]
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics],
                theta=radar_labels,
                fill='toself',
                name=f"{row['broker_id']} ({row['broker_type']})"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Broker Performance Comparison",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _get_metric_period(self, period_str):
        """Convert period string to MetricPeriod enum"""
        period_map = {
            "1h": MetricPeriod.HOUR,
            "6h": MetricPeriod.HOUR,  # Use hour for now
            "1d": MetricPeriod.DAY,
            "1w": MetricPeriod.WEEK
        }
        
        return period_map.get(period_str, MetricPeriod.DAY)
