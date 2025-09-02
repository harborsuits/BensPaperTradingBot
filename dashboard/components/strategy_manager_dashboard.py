#!/usr/bin/env python3
"""
Strategy Manager Dashboard Component

Visualizes strategy performance, active strategies, ensembles,
and provides controls for strategy management.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import logging

logger = logging.getLogger(__name__)

class StrategyManagerDashboard:
    """
    Dashboard component for strategy management and performance visualization.
    """
    
    def __init__(self, app, event_bus, strategy_manager=None):
        """
        Initialize the dashboard component.
        
        Args:
            app: Dash application instance
            event_bus: Event bus for pub/sub communication
            strategy_manager: Reference to the strategy manager (optional)
        """
        self.app = app
        self.event_bus = event_bus
        self.strategy_manager = strategy_manager
        self.component_id = f"strategy-dashboard-{uuid.uuid4()}"
        
        # Register callbacks
        self._register_callbacks()
    
    def layout(self):
        """
        Return the layout for the strategy manager dashboard.
        """
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Strategies Overview"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H4("Active Strategies", className="d-inline"),
                                        html.Span(id="active-strategy-count", className="badge bg-success ms-2")
                                    ]),
                                    html.Hr(),
                                    html.Div(id="active-strategies-table")
                                ], md=12)
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Performance Metrics"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="strategy-performance-graph")
                                ], md=8),
                                dbc.Col([
                                    html.H5("Top Performing Strategies"),
                                    html.Div(id="top-strategies-metrics"),
                                    html.Hr(),
                                    html.H5("Recent Strategy Actions"),
                                    html.Div(id="recent-strategy-actions")
                                ], md=4)
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Ensembles"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="ensemble-list", className="mb-3"),
                                    html.Hr(),
                                    html.Div(id="selected-ensemble-details")
                                ])
                            ])
                        ])
                    ], className="mb-4")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Manager Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        dbc.Button("Start All Strategies", id="btn-start-all", color="success", className="me-2"),
                                        dbc.Button("Stop All Strategies", id="btn-stop-all", color="danger", className="me-2"),
                                        dbc.Button("Evaluate Performance", id="btn-evaluate-perf", color="info")
                                    ], className="mb-3"),
                                    html.Hr(),
                                    html.H5("Strategy Filters"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Asset Type"),
                                            dcc.Dropdown(
                                                id="filter-asset-type",
                                                options=[
                                                    {"label": "All", "value": "all"},
                                                    {"label": "Stocks", "value": "stocks"},
                                                    {"label": "Options", "value": "options"},
                                                    {"label": "Forex", "value": "forex"},
                                                    {"label": "Crypto", "value": "crypto"}
                                                ],
                                                value="all",
                                                clearable=False
                                            )
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Status"),
                                            dcc.Dropdown(
                                                id="filter-status",
                                                options=[
                                                    {"label": "All", "value": "all"},
                                                    {"label": "Running", "value": "running"},
                                                    {"label": "Paused", "value": "paused"},
                                                    {"label": "Stopped", "value": "stopped"},
                                                    {"label": "Error", "value": "error"}
                                                ],
                                                value="all",
                                                clearable=False
                                            )
                                        ], md=6)
                                    ]),
                                    html.Hr(),
                                    html.H5("Risk Parameters"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Max Drawdown Limit"),
                                            dbc.Input(id="input-max-drawdown", type="number", min=0.01, max=0.5, step=0.01, value=0.1)
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Max Allocation per Strategy"),
                                            dbc.Input(id="input-max-alloc-strategy", type="number", min=0.05, max=0.5, step=0.05, value=0.2)
                                        ], md=6)
                                    ])
                                ])
                            ])
                        ])
                    ], className="mb-4")
                ], md=6)
            ]),
            
            # Interval for updates
            dcc.Interval(
                id="strategy-update-interval",
                interval=5000,  # Every 5 seconds
                n_intervals=0
            ),
            
            # Store for active ensemble
            dcc.Store(id="active-ensemble-id")
        ])
    
    def _register_callbacks(self):
        """Register all callbacks for this component."""
        
        # Update active strategies table
        @self.app.callback(
            [Output("active-strategy-count", "children"),
             Output("active-strategies-table", "children")],
            [Input("strategy-update-interval", "n_intervals"),
             Input("filter-asset-type", "value"),
             Input("filter-status", "value")]
        )
        def update_active_strategies(n_intervals, asset_type_filter, status_filter):
            try:
                # If no strategy manager, show placeholder
                if not self.strategy_manager:
                    return "0", html.P("Strategy Manager not connected")
                
                # Get active strategies
                strategies = self.strategy_manager.get_active_strategies()
                filtered_strategies = []
                
                # Apply filters
                for strategy in strategies:
                    # Asset type filter
                    if asset_type_filter != "all" and strategy.get("asset_type") != asset_type_filter:
                        continue
                    
                    # Status filter
                    if status_filter != "all" and strategy.get("state") != status_filter:
                        continue
                    
                    filtered_strategies.append(strategy)
                
                # Create table
                if not filtered_strategies:
                    return "0", html.P("No active strategies match the selected filters")
                
                # Create the table
                table_header = [
                    html.Thead(html.Tr([
                        html.Th("Strategy ID"),
                        html.Th("Name"),
                        html.Th("Asset Type"),
                        html.Th("Symbols"),
                        html.Th("State"),
                        html.Th("Actions")
                    ]))
                ]
                
                table_rows = []
                for strategy in filtered_strategies:
                    # Format symbols for display
                    symbols = strategy.get("symbols", [])
                    if len(symbols) > 3:
                        symbol_text = f"{', '.join(symbols[:2])}... +{len(symbols)-2} more"
                    else:
                        symbol_text = ", ".join(symbols)
                    
                    # Determine state class
                    state = strategy.get("state", "unknown")
                    state_class = {
                        "running": "success",
                        "paused": "warning",
                        "stopped": "secondary",
                        "error": "danger"
                    }.get(state, "info")
                    
                    # Create row
                    row = html.Tr([
                        html.Td(strategy.get("strategy_id", "")),
                        html.Td(strategy.get("name", "")),
                        html.Td(strategy.get("asset_type", "")),
                        html.Td(symbol_text),
                        html.Td(html.Span(state, className=f"badge bg-{state_class}")),
                        html.Td([
                            html.Button("Details", id={"type": "btn-strategy-details", "index": strategy.get("strategy_id")}, 
                                       className="btn btn-sm btn-info me-1"),
                            html.Button("Pause" if state == "running" else "Start", 
                                       id={"type": "btn-strategy-toggle", "index": strategy.get("strategy_id")},
                                       className=f"btn btn-sm {'btn-warning' if state == 'running' else 'btn-success'}")
                        ])
                    ])
                    table_rows.append(row)
                
                table_body = [html.Tbody(table_rows)]
                table = dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True, size="sm")
                
                return str(len(filtered_strategies)), table
                
            except Exception as e:
                logger.error(f"Error updating active strategies: {str(e)}")
                return "Error", html.P(f"Error: {str(e)}")
        
        # Update performance graph
        @self.app.callback(
            Output("strategy-performance-graph", "figure"),
            [Input("strategy-update-interval", "n_intervals")]
        )
        def update_performance_graph(n_intervals):
            try:
                if not self.strategy_manager or not self.strategy_manager.performance_manager:
                    # Create empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Strategy performance data not available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    fig.update_layout(
                        title="Strategy Performance",
                        xaxis_title="Date",
                        yaxis_title="Performance (%)",
                        height=400
                    )
                    return fig
                
                # Get performance metrics for all strategies
                all_metrics = {}
                for strategy_id in self.strategy_manager.strategies:
                    metrics = self.strategy_manager.performance_manager.get_strategy_metrics(strategy_id) or {}
                    if metrics and "daily_returns" in metrics:
                        all_metrics[strategy_id] = metrics
                
                # No metrics data available
                if not all_metrics:
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No performance data available yet",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                    fig.update_layout(
                        title="Strategy Performance",
                        xaxis_title="Date",
                        yaxis_title="Performance (%)",
                        height=400
                    )
                    return fig
                
                # Create performance graph
                fig = go.Figure()
                
                # Add cumulative return line for each strategy
                for strategy_id, metrics in all_metrics.items():
                    strategy = self.strategy_manager.strategies.get(strategy_id)
                    if not strategy:
                        continue
                    
                    # Get daily returns and convert to cumulative
                    returns = metrics.get("daily_returns", {})
                    if not returns:
                        continue
                    
                    # Convert to dataframe
                    df = pd.DataFrame(list(returns.items()), columns=["date", "return"])
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date")
                    
                    # Calculate cumulative returns
                    df["cumulative_return"] = (1 + df["return"]).cumprod() - 1
                    
                    # Add trace
                    fig.add_trace(go.Scatter(
                        x=df["date"],
                        y=df["cumulative_return"] * 100,  # Convert to percentage
                        mode="lines",
                        name=strategy.name
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Strategy Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    height=400,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating performance graph: {str(e)}")
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
        
        # Update top strategies metrics
        @self.app.callback(
            Output("top-strategies-metrics", "children"),
            [Input("strategy-update-interval", "n_intervals")]
        )
        def update_top_strategies(n_intervals):
            try:
                if not self.strategy_manager or not self.strategy_manager.performance_manager:
                    return html.P("Performance data not available")
                
                # Get performance metrics for all strategies
                all_metrics = []
                for strategy_id in self.strategy_manager.strategies:
                    metrics = self.strategy_manager.performance_manager.get_strategy_metrics(strategy_id) or {}
                    if not metrics:
                        continue
                    
                    strategy = self.strategy_manager.strategies.get(strategy_id)
                    if not strategy:
                        continue
                    
                    # Calculate performance score
                    score = metrics.get("sharpe_ratio", 0)
                    if score is None:
                        score = metrics.get("profit_factor", 0)
                    if score is None:
                        score = 0
                    
                    all_metrics.append({
                        "strategy_id": strategy_id,
                        "name": strategy.name,
                        "score": score,
                        "profit_factor": metrics.get("profit_factor", 0),
                        "win_rate": metrics.get("win_rate", 0),
                        "avg_profit": metrics.get("avg_profit", 0),
                        "max_drawdown": metrics.get("max_drawdown", 0)
                    })
                
                # Sort by score and take top 5
                top_strategies = sorted(all_metrics, key=lambda x: x["score"], reverse=True)[:5]
                
                if not top_strategies:
                    return html.P("No performance metrics available yet")
                
                # Create metrics cards
                cards = []
                for strategy in top_strategies:
                    card = dbc.Card([
                        dbc.CardBody([
                            html.H6(strategy["name"], className="card-title"),
                            html.P([
                                html.Span("Win Rate: ", className="fw-bold"),
                                f"{strategy['win_rate']*100:.1f}%"
                            ], className="card-text mb-1"),
                            html.P([
                                html.Span("Profit Factor: ", className="fw-bold"),
                                f"{strategy['profit_factor']:.2f}"
                            ], className="card-text mb-1"),
                            html.P([
                                html.Span("Max Drawdown: ", className="fw-bold"),
                                f"{strategy['max_drawdown']*100:.1f}%"
                            ], className="card-text mb-0")
                        ])
                    ], className="mb-2", style={"border-left": "5px solid #28a745"})
                    cards.append(card)
                
                return html.Div(cards)
                
            except Exception as e:
                logger.error(f"Error updating top strategies: {str(e)}")
                return html.P(f"Error: {str(e)}")
        
        # Update ensemble list
        @self.app.callback(
            Output("ensemble-list", "children"),
            [Input("strategy-update-interval", "n_intervals")]
        )
        def update_ensemble_list(n_intervals):
            try:
                if not self.strategy_manager or not self.strategy_manager.ensembles:
                    return html.P("No strategy ensembles configured")
                
                # Create cards for each ensemble
                cards = []
                for ensemble_id, ensemble in self.strategy_manager.ensembles.items():
                    card = dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.H5(ensemble.name, className="d-inline"),
                                dbc.Badge(f"{len(ensemble.strategies)} Strategies", 
                                        color="info", className="ms-2")
                            ]),
                            html.P(ensemble.description or "No description", className="small text-muted"),
                            dbc.Button("View Details", 
                                    id={"type": "btn-ensemble-details", "index": ensemble_id},
                                    size="sm", color="primary")
                        ])
                    ], className="mb-2", style={"border-left": "5px solid #17a2b8"})
                    cards.append(card)
                
                if not cards:
                    return html.P("No strategy ensembles configured")
                
                return html.Div(cards)
                
            except Exception as e:
                logger.error(f"Error updating ensemble list: {str(e)}")
                return html.P(f"Error: {str(e)}")
        
        # Update ensemble details when clicked
        @self.app.callback(
            [Output("active-ensemble-id", "data"),
             Output("selected-ensemble-details", "children")],
            [Input({"type": "btn-ensemble-details", "index": dash.ALL}, "n_clicks")],
            [State("active-ensemble-id", "data")]
        )
        def update_ensemble_details(btn_clicks, active_ensemble_id):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update
            
            # Get the button that was clicked
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            try:
                clicked_ensemble_id = json.loads(button_id)["index"]
            except:
                return dash.no_update, dash.no_update
            
            try:
                if not self.strategy_manager:
                    return clicked_ensemble_id, html.P("Strategy Manager not connected")
                
                # Get the ensemble
                ensemble = self.strategy_manager.ensembles.get(clicked_ensemble_id)
                if not ensemble:
                    return clicked_ensemble_id, html.P(f"Ensemble {clicked_ensemble_id} not found")
                
                # Create details display
                details = []
                
                # Ensemble info header
                details.append(html.H4(ensemble.name))
                details.append(html.P(ensemble.description or "No description"))
                
                # Ensemble parameters
                param_items = [
                    html.Li([
                        html.Span("Combination Method: ", className="fw-bold"),
                        ensemble.combination_method
                    ]),
                    html.Li([
                        html.Span("Min Consensus: ", className="fw-bold"),
                        f"{ensemble.min_consensus:.2f}"
                    ]),
                    html.Li([
                        html.Span("Auto Adjust Weights: ", className="fw-bold"),
                        str(ensemble.auto_adjust_weights)
                    ])
                ]
                details.append(html.Div([
                    html.H5("Parameters"),
                    html.Ul(param_items)
                ]))
                
                # Strategy weights
                weights_data = []
                for strategy_id, (strategy, weight) in ensemble.strategies.items():
                    weights_data.append({
                        "strategy_id": strategy_id,
                        "name": strategy.name,
                        "weight": weight
                    })
                
                # Create chart for weights
                if weights_data:
                    df = pd.DataFrame(weights_data)
                    fig = px.pie(df, values="weight", names="name", title="Strategy Weights")
                    fig.update_layout(height=300)
                    details.append(dcc.Graph(figure=fig))
                
                return clicked_ensemble_id, html.Div(details)
                
            except Exception as e:
                logger.error(f"Error updating ensemble details: {str(e)}")
                return clicked_ensemble_id, html.P(f"Error: {str(e)}")
        
        # Update recent actions
        @self.app.callback(
            Output("recent-strategy-actions", "children"),
            [Input("strategy-update-interval", "n_intervals")]
        )
        def update_recent_actions(n_intervals):
            if not self.strategy_manager:
                return html.P("Strategy Manager not connected")
            
            # For this example, we'll create some sample actions
            # In a real implementation, you'd get this from the strategy manager's logs
            actions = [
                {"time": "14:25:30", "strategy": "forex_trend_1", "action": "activated", "reason": "Performance above threshold"},
                {"time": "14:15:20", "strategy": "crypto_rsi_1", "action": "weights adjusted", "reason": "Performance improvement"},
                {"time": "13:45:10", "strategy": "stocks_trend_1", "action": "deactivated", "reason": "Max drawdown exceeded"}
            ]
            
            if not actions:
                return html.P("No recent actions")
            
            # Create action list
            action_items = []
            for action in actions:
                item = dbc.ListGroupItem([
                    html.Div([
                        html.Span(action["time"], className="text-muted small me-2"),
                        html.Span(action["strategy"], className="fw-bold")
                    ]),
                    html.Div([
                        html.Span(action["action"].title(), 
                                className=f"badge {'bg-success' if action['action'] == 'activated' else 'bg-warning' if action['action'] == 'weights adjusted' else 'bg-danger'}"),
                        html.Span(f" - {action['reason']}", className="small")
                    ])
                ])
                action_items.append(item)
            
            return dbc.ListGroup(action_items)
        
        # Handle strategy control buttons
        @self.app.callback(
            Output("btn-evaluate-perf", "disabled"),
            [Input("btn-evaluate-perf", "n_clicks")]
        )
        def handle_evaluate_performance(n_clicks):
            if not n_clicks:
                return False
            
            if self.strategy_manager:
                try:
                    # Trigger performance evaluation
                    self.strategy_manager.evaluate_performance()
                    return False
                except Exception as e:
                    logger.error(f"Error evaluating performance: {str(e)}")
            
            return False
