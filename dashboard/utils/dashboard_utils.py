#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Utilities

This module provides utility functions for creating dashboard components.
"""

import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_card(title, value, icon=None, color="primary", id=None):
    """
    Create a card component for displaying metrics.
    
    Args:
        title: Card title
        value: Card value
        icon: Optional FontAwesome icon name
        color: Card color theme (primary, success, warning, danger, info)
        id: Optional component ID
    
    Returns:
        Card component
    """
    icon_element = html.I(className=f"fas fa-{icon} fa-lg mr-2") if icon else None
    
    card_content = [
        dbc.CardHeader(title),
        dbc.CardBody([
            html.Div([
                icon_element,
                html.Span(value, className="h3 ml-2")
            ], className="d-flex align-items-center")
        ])
    ]
    
    return dbc.Card(card_content, className=f"border-{color}", id=id)

def create_alert(title, text, color="info"):
    """
    Create an alert component.
    
    Args:
        title: Alert title
        text: Alert text
        color: Alert color (primary, success, warning, danger, info)
    
    Returns:
        Alert component
    """
    return dbc.Alert(
        [
            html.H6(title, className="alert-heading"),
            html.P(text, className="mb-0")
        ],
        color=color,
        className="mb-3"
    )

def format_number(value, precision=2, as_percentage=False, include_plus=False):
    """
    Format number for display.
    
    Args:
        value: Number to format
        precision: Decimal precision
        as_percentage: Format as percentage
        include_plus: Include + sign for positive values
    
    Returns:
        Formatted string
    """
    # Handle non-numeric values
    if value is None or pd.isna(value):
        return "N/A"
    
    # Convert to percentage if needed
    if as_percentage:
        value = value * 100
    
    # Format number
    if isinstance(value, int):
        formatted = f"{value:,d}"
    else:
        formatted = f"{value:,.{precision}f}"
        if as_percentage:
            formatted += "%"
    
    # Add sign if requested
    if include_plus and value > 0:
        formatted = "+" + formatted
    
    return formatted

def create_time_series_chart(df, x_col, y_cols, title=None, colors=None, height=400):
    """
    Create a time series chart.
    
    Args:
        df: DataFrame with time series data
        x_col: X-axis column (typically datetime)
        y_cols: List of Y-axis columns to plot
        title: Chart title
        colors: List of colors for series
        height: Chart height
    
    Returns:
        Plotly Figure object
    """
    # Default colors if not provided
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each y column
    for i, y_col in enumerate(y_cols):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                name=y_col,
                line=dict(color=color)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_bar_chart(df, x_col, y_col, color_col=None, title=None, height=400):
    """
    Create a bar chart.
    
    Args:
        df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Column to use for color
        title: Chart title
        height: Chart height
    
    Returns:
        Plotly Figure object
    """
    # Create figure
    if color_col:
        fig = go.Figure([
            go.Bar(
                x=df[x_col],
                y=df[y_col],
                marker_color=df[color_col]
            )
        ])
    else:
        fig = go.Figure([
            go.Bar(
                x=df[x_col],
                y=df[y_col]
            )
        ])
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_stats_table(df, title=None):
    """
    Create a statistics table.
    
    Args:
        df: DataFrame with statistics
        title: Table title
    
    Returns:
        Table component
    """
    # Create table header
    header = html.Thead(html.Tr([html.Th(col) for col in df.columns]))
    
    # Create table rows
    rows = []
    for _, row in df.iterrows():
        rows.append(html.Tr([html.Td(row[col]) for col in df.columns]))
    
    # Create table body
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=True, hover=True, responsive=True, striped=True)
    
    # Add title if provided
    if title:
        return html.Div([
            html.H5(title),
            table
        ])
    
    return table
