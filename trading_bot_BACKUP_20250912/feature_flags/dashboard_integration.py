#!/usr/bin/env python3
"""
Feature Flag Dashboard Integration

This module provides a unified dashboard interface for feature flag management,
integrating metrics visualization, A/B testing results, dependency graphs,
and automated rollback monitoring into a single administrative interface.
"""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State

from .service import FeatureFlagService
from .metrics_integration import FeatureFlagMetrics
from .ab_testing import ABTestingManager
from .dependencies_visualizer import DependencyGraph
from .automated_rollbacks import AutomatedRollbackSystem

# Setup logging
logger = logging.getLogger(__name__)

class FeatureFlagDashboard:
    """
    Unified dashboard for feature flag management and visualization.
    
    This class provides a web-based dashboard that integrates:
    - Feature flag status and toggle controls
    - Performance metrics visualization
    - A/B testing results and management
    - Flag dependency visualization
    - Rollback monitoring and alerts
    """
    
    def __init__(self, 
                port: int = 8050,
                feature_flag_service: Optional[FeatureFlagService] = None,
                metrics: Optional[FeatureFlagMetrics] = None,
                ab_testing: Optional[ABTestingManager] = None,
                dependency_graph: Optional[DependencyGraph] = None,
                rollback_system: Optional[AutomatedRollbackSystem] = None):
        """
        Initialize the feature flag dashboard.
        
        Args:
            port: Port to run the dashboard server on
            feature_flag_service: Instance of FeatureFlagService
            metrics: Instance of FeatureFlagMetrics
            ab_testing: Instance of ABTestingManager
            dependency_graph: Instance of DependencyGraph
            rollback_system: Instance of AutomatedRollbackSystem
        """
        self.port = port
        
        # Connect to services or create them if not provided
        self.feature_flag_service = feature_flag_service or FeatureFlagService()
        self.metrics = metrics or FeatureFlagMetrics()
        self.ab_testing = ab_testing or ABTestingManager(metrics_instance=self.metrics)
        self.dependency_graph = dependency_graph or DependencyGraph()
        self.rollback_system = rollback_system or AutomatedRollbackSystem(self.feature_flag_service)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, 
                           title="Feature Flag Management Dashboard",
                           suppress_callback_exceptions=True)
        
        # Setup application layout
        self._setup_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info(f"Feature Flag Dashboard initialized and ready to run on port {port}")
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Feature Flag Management Dashboard"),
                html.P("Monitor, analyze, and control feature flags across the trading platform"),
                html.Hr()
            ], style={'margin-bottom': '20px'}),
            
            # Navigation tabs
            dcc.Tabs(id='tabs', value='tab-overview', children=[
                dcc.Tab(label='Overview', value='tab-overview'),
                dcc.Tab(label='Feature Flags', value='tab-flags'),
                dcc.Tab(label='A/B Testing', value='tab-ab-testing'),
                dcc.Tab(label='Metrics & Impact', value='tab-metrics'),
                dcc.Tab(label='Dependencies', value='tab-dependencies'),
                dcc.Tab(label='Rollback Monitoring', value='tab-rollbacks'),
            ]),
            
            # Tab content
            html.Div(id='tab-content'),
            
            # Hidden divs for storing state
            html.Div(id='refresh-trigger', style={'display': 'none'}),
            dcc.Interval(id='interval-component', interval=30*1000, n_intervals=0),  # 30-second refresh
            
            # Notification area
            html.Div(id='notification-area', className='notification-area'),
            
            # Footer
            html.Footer([
                html.Hr(),
                html.P(f"Feature Flag Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            ])
        ])
    
    def _register_callbacks(self):
        """Register all dashboard callbacks"""
        # Main tab navigation callback
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('tabs', 'value'), Input('refresh-trigger', 'children')]
        )
        def render_tab_content(tab, _):
            """Render content based on selected tab"""
            if tab == 'tab-overview':
                return self._render_overview_tab()
            elif tab == 'tab-flags':
                return self._render_flags_tab()
            elif tab == 'tab-ab-testing':
                return self._render_ab_testing_tab()
            elif tab == 'tab-metrics':
                return self._render_metrics_tab()
            elif tab == 'tab-dependencies':
                return self._render_dependencies_tab()
            elif tab == 'tab-rollbacks':
                return self._render_rollbacks_tab()
            return html.Div("Tab content not found")
        
        # Toggle feature flag callback
        @self.app.callback(
            [Output('notification-area', 'children'),
             Output('refresh-trigger', 'children')],
            [Input('toggle-flag-button', 'n_clicks')],
            [State('flag-dropdown', 'value'),
             State('toggle-value', 'value')]
        )
        def toggle_feature_flag(n_clicks, flag_name, enabled):
            if n_clicks is None or not flag_name:
                return None, datetime.now().isoformat()
            
            try:
                if enabled:
                    self.feature_flag_service.enable_flag(flag_name)
                    message = f"Flag '{flag_name}' has been enabled"
                else:
                    self.feature_flag_service.disable_flag(flag_name)
                    message = f"Flag '{flag_name}' has been disabled"
                
                notification = html.Div(message, className='notification success')
                return notification, datetime.now().isoformat()
            except Exception as e:
                notification = html.Div(f"Error: {str(e)}", className='notification error')
                return notification, datetime.now().isoformat()
        
        # Create experiment callback
        @self.app.callback(
            [Output('experiment-notification', 'children'),
             Output('refresh-trigger', 'children', allow_duplicate=True)],
            [Input('create-experiment-button', 'n_clicks')],
            [State('experiment-name', 'value'),
             State('experiment-flag', 'value'),
             State('experiment-hypothesis', 'value')]
        )
        def create_experiment(n_clicks, name, flag_name, hypothesis):
            if n_clicks is None or not name or not flag_name:
                return None, datetime.now().isoformat()
            
            try:
                self.ab_testing.create_experiment(
                    name=name,
                    flag_name=flag_name,
                    hypothesis=hypothesis or "",
                    min_sample_size=1000,
                    max_duration_days=14
                )
                
                message = f"Experiment '{name}' for flag '{flag_name}' has been created"
                notification = html.Div(message, className='notification success')
                return notification, datetime.now().isoformat()
            except Exception as e:
                notification = html.Div(f"Error: {str(e)}", className='notification error')
                return notification, datetime.now().isoformat()
        
        # Auto-refresh on interval
        @self.app.callback(
            Output('refresh-trigger', 'children', allow_duplicate=True),
            [Input('interval-component', 'n_intervals')]
        )
        def refresh_data(_):
            return datetime.now().isoformat()
    
    def _render_overview_tab(self) -> html.Div:
        """Render the overview dashboard tab"""
        # Get flag statistics
        flag_stats = self.metrics.get_usage_statistics()
        
        # Get active experiments
        active_experiments = self.ab_testing.get_running_experiments()
        
        # Get alerts from rollback system
        active_alerts = self.rollback_system.get_active_alerts() if hasattr(self.rollback_system, 'get_active_alerts') else []
        
        # Create summary cards
        flag_count = len(self.feature_flag_service.get_all_flags())
        enabled_count = len([f for f in self.feature_flag_service.get_all_flags() if f.enabled])
        
        overview_div = html.Div([
            # Summary statistics
            html.Div([
                html.Div([
                    html.H3(flag_count),
                    html.P("Total Feature Flags")
                ], className='stat-card'),
                
                html.Div([
                    html.H3(enabled_count),
                    html.P("Enabled Flags")
                ], className='stat-card'),
                
                html.Div([
                    html.H3(len(active_experiments)),
                    html.P("Active Experiments")
                ], className='stat-card'),
                
                html.Div([
                    html.H3(len(active_alerts)),
                    html.P("Active Alerts"),
                    html.Span(len(active_alerts), className='alert-badge' if len(active_alerts) > 0 else 'hidden')
                ], className='stat-card' + (' alert' if len(active_alerts) > 0 else ''))
            ], className='stat-container'),
            
            # Charts row
            html.Div([
                # Flag usage chart
                html.Div([
                    html.H3("Top Flag Usage"),
                    dcc.Graph(figure=self._create_flag_usage_chart(flag_stats))
                ], className='chart-card wide'),
                
                # Latest evaluations chart
                html.Div([
                    html.H3("Flag Evaluations (24h)"),
                    dcc.Graph(figure=self._create_evaluations_chart())
                ], className='chart-card')
            ], className='chart-container'),
            
            # Recent activity and alerts
            html.Div([
                html.Div([
                    html.H3("Recent Activity"),
                    self._render_activity_feed()
                ], className='card'),
                
                html.Div([
                    html.H3("Active Alerts"),
                    self._render_alerts_list(active_alerts)
                ], className='card')
            ], className='card-container')
        ])
        
        return overview_div
    
    def _render_flags_tab(self) -> html.Div:
        """Render the feature flags management tab"""
        # Get all flags
        all_flags = self.feature_flag_service.get_all_flags()
        
        # Create flag management UI
        flags_div = html.Div([
            html.H2("Feature Flags Management"),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Select Flag:"),
                    dcc.Dropdown(
                        id='flag-dropdown',
                        options=[{'label': f.name, 'value': f.name} for f in all_flags],
                        value=all_flags[0].name if all_flags else None
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Enabled:"),
                    dcc.RadioItems(
                        id='toggle-value',
                        options=[
                            {'label': 'Enabled', 'value': True},
                            {'label': 'Disabled', 'value': False}
                        ],
                        value=all_flags[0].enabled if all_flags else False,
                        inline=True
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Button('Apply Changes', id='toggle-flag-button', className='button primary')
            ], className='control-panel'),
            
            # Flags table
            html.Div([
                html.H3("All Feature Flags"),
                self._render_flags_table(all_flags)
            ], className='table-container'),
            
            # Flag details
            html.Div([
                html.H3("Flag Details"),
                html.Div(id='flag-details')
            ], className='details-panel')
        ])
        
        return flags_div
    
    def _render_ab_testing_tab(self) -> html.Div:
        """Render the A/B testing management tab"""
        # Get all experiments
        experiments = list(self.ab_testing.experiments.values())
        
        # Get all flags for dropdown
        all_flags = self.feature_flag_service.get_all_flags()
        
        ab_testing_div = html.Div([
            html.H2("A/B Testing"),
            
            # Create experiment form
            html.Div([
                html.H3("Create New Experiment"),
                
                html.Div([
                    html.Div([
                        html.Label("Experiment Name:"),
                        dcc.Input(id='experiment-name', type='text', placeholder='Enter experiment name')
                    ], className='form-group'),
                    
                    html.Div([
                        html.Label("Feature Flag:"),
                        dcc.Dropdown(
                            id='experiment-flag',
                            options=[{'label': f.name, 'value': f.name} for f in all_flags],
                            placeholder='Select a feature flag'
                        )
                    ], className='form-group'),
                    
                    html.Div([
                        html.Label("Hypothesis:"),
                        dcc.Textarea(
                            id='experiment-hypothesis',
                            placeholder='What do you expect this experiment to demonstrate?',
                            style={'width': '100%', 'height': '100px'}
                        )
                    ], className='form-group'),
                    
                    html.Button('Create Experiment', id='create-experiment-button', className='button primary'),
                    html.Div(id='experiment-notification')
                ], className='form-container')
            ], className='card'),
            
            # Active experiments
            html.Div([
                html.H3("Active Experiments"),
                self._render_experiments_table([e for e in experiments if hasattr(e, 'status') and e.status.value == 'running'])
            ], className='table-container'),
            
            # Completed experiments
            html.Div([
                html.H3("Completed Experiments"),
                self._render_experiments_table([e for e in experiments if hasattr(e, 'status') and e.status.value in ['completed', 'inconclusive']])
            ], className='table-container')
        ])
        
        return ab_testing_div
    
    def _render_metrics_tab(self) -> html.Div:
        """Render the metrics and impact analysis tab"""
        # Get flags with impact data
        flags_with_metrics = []
        for flag_name in self.metrics.performance_samples.keys():
            if len(self.metrics.performance_samples[flag_name]) > 20:
                flags_with_metrics.append(flag_name)
        
        metrics_div = html.Div([
            html.H2("Metrics & Impact Analysis"),
            
            # Flag metrics selection
            html.Div([
                html.Label("Select Flag:"),
                dcc.Dropdown(
                    id='metrics-flag-dropdown',
                    options=[{'label': name, 'value': name} for name in flags_with_metrics],
                    value=flags_with_metrics[0] if flags_with_metrics else None
                )
            ], className='control-panel'),
            
            # Metrics visualization
            html.Div([
                html.H3("Performance Impact"),
                dcc.Graph(id='metrics-impact-graph')
            ], className='chart-card wide'),
            
            # Detailed metrics table
            html.Div([
                html.H3("Detailed Metrics"),
                html.Div(id='metrics-table')
            ], className='table-container')
        ])
        
        return metrics_div
    
    def _render_dependencies_tab(self) -> html.Div:
        """Render the dependencies visualization tab"""
        dependencies_div = html.Div([
            html.H2("Flag Dependencies"),
            
            # Controls
            html.Div([
                html.Button('Generate Interactive Graph', id='generate-graph-button', className='button primary'),
                html.Button('Export as DOT File', id='export-dot-button', className='button secondary'),
                html.Div(id='dependency-notification')
            ], className='control-panel'),
            
            # Dependencies visualization
            html.Div([
                html.H3("Dependency Graph"),
                dcc.Graph(id='dependency-graph')
            ], className='chart-card full-width'),
            
            # Dependencies table
            html.Div([
                html.H3("All Dependencies"),
                self._render_dependencies_table()
            ], className='table-container')
        ])
        
        return dependencies_div
    
    def _render_rollbacks_tab(self) -> html.Div:
        """Render the rollback monitoring tab"""
        # Get rules and alerts
        rollback_rules = self.rollback_system.get_all_rules() if hasattr(self.rollback_system, 'get_all_rules') else []
        active_alerts = self.rollback_system.get_active_alerts() if hasattr(self.rollback_system, 'get_active_alerts') else []
        
        rollbacks_div = html.Div([
            html.H2("Rollback Monitoring"),
            
            # Alert summary
            html.Div([
                html.H3("Active Alerts"),
                self._render_alerts_list(active_alerts)
            ], className='card'),
            
            # Rules table
            html.Div([
                html.H3("Rollback Rules"),
                self._render_rules_table(rollback_rules)
            ], className='table-container'),
            
            # Monitoring dashboard
            html.Div([
                html.H3("Monitoring Dashboard"),
                dcc.Graph(id='monitoring-graph')
            ], className='chart-card full-width')
        ])
        
        return rollbacks_div
    
    def _create_flag_usage_chart(self, flag_stats: Dict[str, Any]) -> go.Figure:
        """Create a bar chart showing flag usage"""
        if not flag_stats:
            return go.Figure().add_annotation(
                text="No flag usage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Extract top 10 flags by total evaluations
        top_flags = sorted(flag_stats.items(), key=lambda x: x[1]['total_evaluations'], reverse=True)[:10]
        
        flag_names = [f[0] for f in top_flags]
        true_counts = [f[1]['true_count'] for f in top_flags]
        false_counts = [f[1]['false_count'] for f in top_flags]
        
        fig = go.Figure(data=[
            go.Bar(name='Enabled', x=flag_names, y=true_counts, marker_color='#2ca02c'),
            go.Bar(name='Disabled', x=flag_names, y=false_counts, marker_color='#d62728')
        ])
        
        fig.update_layout(
            barmode='stack',
            xaxis_title='Flag Name',
            yaxis_title='Number of Evaluations',
            legend_title='Flag Value',
            hovermode='closest'
        )
        
        return fig
    
    def _create_evaluations_chart(self) -> go.Figure:
        """Create a time series chart showing flag evaluations over time"""
        # Sample data - in a real implementation, this would come from time-series metrics
        # For now, we'll create synthetic data
        now = datetime.now()
        hours = list(range(24))
        times = [(now - timedelta(hours=h)).strftime("%H:00") for h in hours]
        times.reverse()  # Oldest to newest
        
        # Create random-ish data for demo
        values = [int(50 + 30 * np.sin(h/2) + 20 * np.random.random()) for h in hours]
        values.reverse()  # Match times order
        
        fig = go.Figure(data=go.Scatter(
            x=times, 
            y=values,
            mode='lines+markers',
            name='Evaluations',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Evaluations per Hour',
            hovermode='x unified'
        )
        
        return fig
    
    def _render_activity_feed(self) -> html.Div:
        """Render the recent activity feed"""
        # In a real implementation, this would show actual recent events
        # For now, we'll use placeholder data
        activities = [
            {"time": "14:32", "event": "Flag 'advanced_risk_controls' enabled by admin"},
            {"time": "13:45", "event": "Experiment 'ui_redesign_test' completed with significant results"},
            {"time": "11:30", "event": "New flag 'algorithmic_trading_v2' created"},
            {"time": "09:15", "event": "Auto-rollback triggered for 'real_time_quotes'"},
        ]
        
        activity_items = [
            html.Div([
                html.Span(activity["time"], className='time'),
                html.Span(activity["event"])
            ], className='activity-item')
            for activity in activities
        ]
        
        return html.Div(activity_items, className='activity-feed')
    
    def _render_alerts_list(self, alerts: List[Dict[str, Any]]) -> html.Div:
        """Render a list of active alerts"""
        if not alerts:
            return html.Div("No active alerts", className='empty-state')
        
        alert_items = [
            html.Div([
                html.Div([
                    html.Span(alert.get("severity", "Unknown"), className=f'severity {alert.get("severity", "").lower()}'),
                    html.H4(alert.get("title", "Alert")),
                ], className='alert-header'),
                html.P(alert.get("description", "")),
                html.Div([
                    html.Span(f"Flag: {alert.get('flag_name', 'Unknown')}"),
                    html.Span(f"Time: {alert.get('created_at', 'Unknown')}")
                ], className='alert-footer')
            ], className=f'alert-item {alert.get("severity", "").lower()}')
            for alert in alerts
        ]
        
        return html.Div(alert_items, className='alerts-list')
    
    def _render_flags_table(self, flags: List[Any]) -> html.Table:
        """Render a table of feature flags"""
        if not flags:
            return html.Div("No feature flags found", className='empty-state')
        
        # Create header row
        header = html.Tr([
            html.Th("Name"),
            html.Th("Description"),
            html.Th("Category"),
            html.Th("Status"),
            html.Th("Created"),
            html.Th("Modified")
        ])
        
        # Create rows for each flag
        rows = [
            html.Tr([
                html.Td(flag.name),
                html.Td(flag.description if hasattr(flag, 'description') else ""),
                html.Td(flag.category.value if hasattr(flag, 'category') else ""),
                html.Td(html.Span("Enabled", className='status enabled') if flag.enabled else 
                       html.Span("Disabled", className='status disabled')),
                html.Td(flag.created_at.split("T")[0] if hasattr(flag, 'created_at') else ""),
                html.Td(flag.last_modified.split("T")[0] if hasattr(flag, 'last_modified') else "")
            ])
            for flag in flags
        ]
        
        return html.Table([
            html.Thead(header),
            html.Tbody(rows)
        ], className='data-table')
    
    def _render_experiments_table(self, experiments: List[Any]) -> html.Table:
        """Render a table of A/B test experiments"""
        if not experiments:
            return html.Div("No experiments found", className='empty-state')
        
        # Create header row
        header = html.Tr([
            html.Th("Name"),
            html.Th("Flag"),
            html.Th("Status"),
            html.Th("Sample Size"),
            html.Th("Start Date"),
            html.Th("Actions")
        ])
        
        # Create rows for each experiment
        rows = [
            html.Tr([
                html.Td(exp.name),
                html.Td(exp.flag_name),
                html.Td(html.Span(exp.status.value.capitalize(), className=f'status {exp.status.value}')),
                html.Td(f"{sum(exp.current_sample_size.values())} / {exp.min_sample_size}"),
                html.Td(exp.started_at.split("T")[0] if exp.started_at else "Not started"),
                html.Td(html.Button("View Results", className='button small'))
            ])
            for exp in experiments
        ]
        
        return html.Table([
            html.Thead(header),
            html.Tbody(rows)
        ], className='data-table')
    
    def _render_dependencies_table(self) -> html.Table:
        """Render a table of flag dependencies"""
        # Get all edges from dependency graph
        if not hasattr(self.dependency_graph, 'graph') or not self.dependency_graph.graph.edges:
            return html.Div("No dependencies defined", className='empty-state')
        
        edges = list(self.dependency_graph.graph.edges(data=True))
        
        if not edges:
            return html.Div("No dependencies defined", className='empty-state')
        
        # Create header row
        header = html.Tr([
            html.Th("Source Flag"),
            html.Th("Relationship"),
            html.Th("Target Flag"),
            html.Th("Description")
        ])
        
        # Create rows for each dependency
        rows = [
            html.Tr([
                html.Td(source),
                html.Td(data.get('type', 'unknown')),
                html.Td(target),
                html.Td(data.get('description', ''))
            ])
            for source, target, data in edges
        ]
        
        return html.Table([
            html.Thead(header),
            html.Tbody(rows)
        ], className='data-table')
    
    def _render_rules_table(self, rules: List[Dict[str, Any]]) -> html.Table:
        """Render a table of rollback rules"""
        if not rules:
            return html.Div("No rollback rules defined", className='empty-state')
        
        # Create header row
        header = html.Tr([
            html.Th("Rule Name"),
            html.Th("Flag"),
            html.Th("Conditions"),
            html.Th("Strategy"),
            html.Th("Status"),
            html.Th("Actions")
        ])
        
        # Create rows for each rule
        rows = [
            html.Tr([
                html.Td(rule.get('name', 'Unnamed')),
                html.Td(rule.get('flag_name', 'Unknown')),
                html.Td(f"{len(rule.get('conditions', []))} condition(s)"),
                html.Td(rule.get('strategy', 'Unknown')),
                html.Td(html.Span(
                    "Active" if rule.get('active', False) else "Inactive", 
                    className=f'status {"enabled" if rule.get("active", False) else "disabled"}'
                )),
                html.Td(html.Button("Edit", className='button small'))
            ])
            for rule in rules
        ]
        
        return html.Table([
            html.Thead(header),
            html.Tbody(rows)
        ], className='data-table')
    
    def run(self, debug: bool = False):
        """
        Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting Feature Flag Dashboard on port {self.port}")
        self.app.run_server(port=self.port, debug=debug)


def create_dashboard(config_path: Optional[str] = None) -> FeatureFlagDashboard:
    """
    Create and configure a feature flag dashboard with default services.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured FeatureFlagDashboard instance
    """
    # Create service instances
    feature_flag_service = FeatureFlagService(config_path=config_path)
    metrics = FeatureFlagMetrics()
    ab_testing = ABTestingManager(metrics_instance=metrics)
    dependency_graph = DependencyGraph()
    rollback_system = AutomatedRollbackSystem(feature_flag_service)
    
    # Create and return dashboard
    return FeatureFlagDashboard(
        feature_flag_service=feature_flag_service,
        metrics=metrics,
        ab_testing=ab_testing,
        dependency_graph=dependency_graph,
        rollback_system=rollback_system
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dashboard = create_dashboard()
    dashboard.run(debug=True) 