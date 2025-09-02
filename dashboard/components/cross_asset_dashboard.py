#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Asset Dashboard Component

This module implements dashboard components to visualize cross-asset opportunities,
exceptional trading signals, and adaptive strategy performance.
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
import json
import os

# Import trading system components
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.core.cross_asset_opportunity_ranker import CrossAssetOpportunityRanker
from trading_bot.core.capital_allocation_optimizer import CapitalAllocationOptimizer
from trading_bot.core.cross_asset_risk_manager import CrossAssetRiskManager
from trading_bot.core.strategy_performance_feedback import StrategyPerformanceFeedback

# Dashboard utilities
from dashboard.utils.dashboard_utils import create_card, format_number, create_alert

class CrossAssetDashboard:
    """
    Cross-Asset Dashboard Component
    
    This class implements dashboard components for the enhanced trading system:
    - Cross-asset opportunity visualization
    - Exceptional opportunity alerts
    - Strategy performance feedback visualization
    - Capital allocation and risk management insights
    """
    
    def __init__(self, app, event_bus):
        """
        Initialize Cross-Asset Dashboard.
        
        Args:
            app: Dash application instance
            event_bus: Event bus instance
        """
        self.app = app
        self.event_bus = event_bus
        
        # Initialize state
        self.opportunities = []
        self.allocations = {}
        self.risk_metrics = {}
        self.strategy_performance = {}
        self.exceptional_opportunities = []
        
        # Register event handlers
        self.event_bus.subscribe(EventType.OPPORTUNITIES_RANKED, self._on_opportunities_ranked)
        self.event_bus.subscribe(EventType.CAPITAL_ALLOCATIONS_UPDATED, self._on_allocations_updated)
        self.event_bus.subscribe(EventType.RISK_METRICS_UPDATED, self._on_risk_metrics_updated)
        self.event_bus.subscribe(EventType.STRATEGY_PERFORMANCE_WEIGHTS, self._on_strategy_performance)
        
        # Register callback for tab selection
        self._register_callbacks()
    
    def _on_opportunities_ranked(self, event):
        """Handle opportunities ranked events."""
        if 'opportunities' in event.data:
            opportunities = event.data['opportunities']
            
            # Update opportunities state
            self.opportunities = opportunities.get('top_opportunities', [])
            self.exceptional_opportunities = opportunities.get('exceptional_opportunities', [])
    
    def _on_allocations_updated(self, event):
        """Handle capital allocations updated events."""
        if 'allocations' in event.data:
            self.allocations = event.data['allocations']
    
    def _on_risk_metrics_updated(self, event):
        """Handle risk metrics updated events."""
        if 'risk_metrics' in event.data:
            self.risk_metrics = event.data['risk_metrics']
            self.risk_warnings = event.data.get('warnings', [])
            self.risk_exposures = event.data.get('exposures', {})
    
    def _on_strategy_performance(self, event):
        """Handle strategy performance events."""
        if 'strategy_weights' in event.data:
            self.strategy_performance = {
                'weights': event.data['strategy_weights'],
                'regime': event.data['market_regime'],
                'performance': event.data.get('performance_data', {})
            }
    
    def _register_callbacks(self):
        """Register dashboard callbacks."""
        # Update opportunity table
        self.app.callback(
            Output('cross-asset-opportunity-table', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_opportunity_table)
        
        # Update exceptional opportunities
        self.app.callback(
            Output('exceptional-opportunities-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_exceptional_opportunities)
        
        # Update allocation chart
        self.app.callback(
            Output('allocation-chart-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_allocation_chart)
        
        # Update risk metrics
        self.app.callback(
            Output('risk-metrics-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_risk_metrics)
        
        # Update strategy performance
        self.app.callback(
            Output('strategy-performance-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_strategy_performance)
    
    def layout(self):
        """Create dashboard layout."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3("Cross-Asset Opportunities", className="mb-4"),
                    dbc.Card([
                        dbc.CardHeader("Top Trading Opportunities"),
                        dbc.CardBody(id="cross-asset-opportunity-table")
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Exceptional Opportunities", className="bg-success text-white"),
                        dbc.CardBody(id="exceptional-opportunities-container")
                    ], className="mb-4")
                ], md=8),
                
                dbc.Col([
                    html.H3("Capital Allocation", className="mb-4"),
                    dbc.Card([
                        dbc.CardHeader("Asset Allocation"),
                        dbc.CardBody(id="allocation-chart-container")
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody(id="risk-metrics-container")
                    ], className="mb-4")
                ], md=4)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Strategy Performance", className="mb-4"),
                    dbc.Card([
                        dbc.CardHeader("Strategy Performance by Market Regime"),
                        dbc.CardBody(id="strategy-performance-container")
                    ])
                ])
            ])
        ])
    
    def update_opportunity_table(self, n_intervals):
        """Update opportunity table."""
        if not self.opportunities:
            return html.Div("No trading opportunities available")
        
        # Create DataFrame for table
        df = pd.DataFrame(self.opportunities)
        
        # Format table
        table_header = [
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Asset Class"),
                html.Th("Signal Type"),
                html.Th("Score"),
                html.Th("Expected Return"),
                html.Th("Time Horizon")
            ]))
        ]
        
        rows = []
        for opp in self.opportunities:
            # Determine row color based on score
            if opp['score'] >= 85:
                row_class = "bg-success text-white"
            elif opp['score'] >= 75:
                row_class = "bg-info text-white"
            elif opp['score'] >= 65:
                row_class = "bg-light"
            else:
                row_class = ""
            
            # Create row
            row = html.Tr([
                html.Td(opp['symbol']),
                html.Td(opp['asset_class'].upper()),
                html.Td(opp['signal_type']),
                html.Td(f"{opp['score']:.1f}"),
                html.Td(f"{opp.get('expected_return', 0):.2f}%"),
                html.Td(opp.get('time_horizon', 'medium').capitalize())
            ], className=row_class)
            
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)
    
    def update_exceptional_opportunities(self, n_intervals):
        """Update exceptional opportunities."""
        if not self.exceptional_opportunities:
            return html.Div("No exceptional opportunities available")
        
        cards = []
        for opp in self.exceptional_opportunities:
            # Create card for exceptional opportunity
            card = dbc.Card([
                dbc.CardHeader(f"{opp['symbol']} - {opp['asset_class'].upper()}", className="bg-success text-white"),
                dbc.CardBody([
                    html.H5(f"Opportunity Score: {opp['score']:.1f}", className="card-title"),
                    html.P([
                        html.Span("This is an exceptional opportunity that deserves immediate attention!"),
                        html.Br(),
                        html.Small(f"Generated at {datetime.now().strftime('%H:%M:%S')}")
                    ])
                ])
            ], className="mb-3")
            
            cards.append(card)
        
        return html.Div(cards)
    
    def update_allocation_chart(self, n_intervals):
        """Update allocation chart."""
        if not self.allocations:
            return html.Div("No allocation data available")
        
        # Get asset class allocations
        asset_class_percentages = self.allocations.get('asset_class_percentages', {})
        
        if not asset_class_percentages:
            return html.Div("No asset allocation data available")
        
        # Create DataFrame for pie chart
        df = pd.DataFrame([
            {"Asset Class": asset.upper(), "Allocation": pct * 100}
            for asset, pct in asset_class_percentages.items()
            if pct > 0
        ])
        
        # Create pie chart
        if not df.empty:
            fig = px.pie(
                df, 
                values="Allocation", 
                names="Asset Class", 
                title="Current Asset Allocation",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            chart = dcc.Graph(figure=fig)
        else:
            chart = html.Div("No allocation data")
        
        # Add total allocated info
        total_allocated = self.allocations.get('total_allocated_percent', 0) * 100
        remaining_capital = self.allocations.get('remaining_capital', 0)
        
        info = html.Div([
            html.P(f"Total Allocated: {total_allocated:.1f}%"),
            html.P(f"Remaining Capital: ${remaining_capital:,.2f}")
        ])
        
        return html.Div([chart, info])
    
    def update_risk_metrics(self, n_intervals):
        """Update risk metrics."""
        if not self.risk_metrics:
            return html.Div("No risk metrics available")
        
        # Create risk metrics cards
        cards = []
        
        # Portfolio variance
        portfolio_var = self.risk_metrics.get('portfolio_var', 0)
        var_card = create_card(
            "Portfolio Variance",
            f"{portfolio_var * 100:.2f}%",
            icon="chart-line"
        )
        cards.append(dbc.Col(var_card, md=6))
        
        # 95% VaR
        var_95 = self.risk_metrics.get('var_95', 0)
        var_95_card = create_card(
            "95% VaR",
            f"${var_95:,.2f}",
            icon="exclamation-triangle",
            color="warning"
        )
        cards.append(dbc.Col(var_95_card, md=6))
        
        # Risk warnings
        warnings_content = []
        for warning in self.risk_warnings:
            warning_type = warning.get('type', '')
            severity = warning.get('severity', 'medium')
            
            if severity == 'high':
                color = "danger"
            elif severity == 'medium':
                color = "warning"
            else:
                color = "info"
            
            # Create alert for warning
            if warning_type == 'currency_concentration':
                alert = create_alert(
                    f"High {warning['currency']} Exposure: {warning['exposure'] * 100:.1f}%",
                    f"Currency exposure exceeds {warning['limit'] * 100:.1f}% limit",
                    color
                )
            elif warning_type == 'sector_concentration':
                alert = create_alert(
                    f"High {warning['sector']} Sector Exposure: {warning['exposure'] * 100:.1f}%",
                    f"Sector exposure exceeds {warning['limit'] * 100:.1f}% limit",
                    color
                )
            elif warning_type == 'portfolio_variance':
                alert = create_alert(
                    f"High Portfolio Variance: {warning['variance'] * 100:.2f}%",
                    f"Portfolio variance exceeds {warning['limit'] * 100:.2f}% limit",
                    color
                )
            else:
                alert = create_alert(
                    f"Risk Warning: {warning_type}",
                    "Unknown risk warning",
                    color
                )
            
            warnings_content.append(alert)
        
        # If no warnings, show "All Clear"
        if not warnings_content:
            warnings_content = [create_alert(
                "All Clear",
                "No risk warnings detected",
                "success"
            )]
        
        # Add cards and warnings to layout
        return html.Div([
            dbc.Row(cards, className="mb-4"),
            html.H5("Risk Warnings"),
            html.Div(warnings_content)
        ])
    
    def update_strategy_performance(self, n_intervals):
        """Update strategy performance."""
        if not self.strategy_performance:
            return html.Div("No strategy performance data available")
        
        # Get current regime
        current_regime = self.strategy_performance.get('regime', 'unknown')
        
        # Get strategy weights
        weights = self.strategy_performance.get('weights', {})
        
        if not weights:
            return html.Div("No strategy weight data available")
        
        # Create DataFrame for bar chart
        df = pd.DataFrame([
            {"Strategy": strategy, "Weight": weight * 100}
            for strategy, weight in weights.items()
            if weight > 0
        ])
        
        # Sort by weight
        if not df.empty:
            df = df.sort_values("Weight", ascending=False)
        
        # Create bar chart
        if not df.empty:
            fig = px.bar(
                df, 
                x="Strategy", 
                y="Weight",
                title=f"Strategy Weights ({current_regime.capitalize()} Regime)",
                color="Weight",
                color_continuous_scale="Viridis"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Strategy",
                yaxis_title="Weight (%)",
                height=400
            )
            
            chart = dcc.Graph(figure=fig)
        else:
            chart = html.Div("No strategy weight data")
        
        # Get performance data
        performance = self.strategy_performance.get('performance', {})
        
        # Create performance summary
        if performance:
            total_trades = performance.get('total_trades', 0)
            win_rate = performance.get('win_rate', 0) * 100
            profit_factor = performance.get('profit_factor', 0)
            
            summary = html.Div([
                html.H5(f"Regime Performance: {current_regime.capitalize()}"),
                html.P([
                    f"Total Trades: {total_trades} | ",
                    f"Win Rate: {win_rate:.1f}% | ",
                    f"Profit Factor: {profit_factor:.2f}"
                ])
            ], className="mt-3")
        else:
            summary = html.Div("No performance data")
        
        return html.Div([chart, summary])


# Custom styles for exceptional opportunities
exceptional_opportunity_style = {
    'border': '2px solid #28a745',
    'boxShadow': '0 0 10px rgba(40, 167, 69, 0.5)',
    'animation': 'pulse 2s infinite'
}

# Add custom CSS for animations
custom_css = '''
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
    100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
}
'''
