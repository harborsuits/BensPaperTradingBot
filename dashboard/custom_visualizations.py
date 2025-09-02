"""
Customizable visualization components for the BensBot Trading Dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple, Union

from dashboard.theme import COLORS

# Theme presets for charts
CHART_THEMES = {
    "default": {
        "background": "white",
        "gridcolor": "#f0f0f0",
        "primary_color": COLORS["primary"],
        "secondary_color": COLORS["secondary"],
        "accent_colors": [COLORS["primary"], COLORS["secondary"], COLORS["info"], 
                         COLORS["success"], COLORS["warning"], COLORS["danger"]],
        "font_family": "Arial, sans-serif",
        "title_font_size": 18,
        "axis_font_size": 12,
        "label_font_size": 10
    },
    "dark": {
        "background": "#1f2937",
        "gridcolor": "#374151",
        "primary_color": "#3b82f6",
        "secondary_color": "#8b5cf6",
        "accent_colors": ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"],
        "font_family": "Arial, sans-serif",
        "title_font_size": 18,
        "axis_font_size": 12,
        "label_font_size": 10
    },
    "institutional": {
        "background": "#0a0e17",
        "gridcolor": "#1f2937",
        "primary_color": "#4f80bd",
        "secondary_color": "#5b8c85",
        "accent_colors": ["#4f80bd", "#5b8c85", "#a4c2f4", "#b6d7a8", "#ffe599", "#ea9999"],
        "font_family": "Arial, sans-serif",
        "title_font_size": 16,
        "axis_font_size": 12,
        "label_font_size": 10
    },
    "minimal": {
        "background": "white",
        "gridcolor": "#eee",
        "primary_color": "#555",
        "secondary_color": "#999",
        "accent_colors": ["#555", "#777", "#999", "#bbb", "#ddd", "#efefef"],
        "font_family": "Arial, sans-serif",
        "title_font_size": 14,
        "axis_font_size": 10,
        "label_font_size": 8
    }
}

class ChartCustomizer:
    """Customizer for Plotly charts"""
    
    @staticmethod
    def apply_theme(fig: go.Figure, theme_name: str = "default") -> go.Figure:
        """Apply a theme to a Plotly figure"""
        if theme_name not in CHART_THEMES:
            theme_name = "default"
            
        theme = CHART_THEMES[theme_name]
        
        # Apply background colors
        fig.update_layout(
            plot_bgcolor=theme["background"],
            paper_bgcolor=theme["background"],
            font=dict(
                family=theme["font_family"],
                size=theme["axis_font_size"],
                color="#fff" if theme_name in ["dark", "institutional"] else "#333"
            ),
            title=dict(
                font=dict(
                    family=theme["font_family"],
                    size=theme["title_font_size"],
                    color="#fff" if theme_name in ["dark", "institutional"] else "#333"
                )
            )
        )
        
        # Update grid color
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=theme["gridcolor"],
            zerolinecolor=theme["gridcolor"]
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=theme["gridcolor"],
            zerolinecolor=theme["gridcolor"]
        )
        
        # Update trace colors for better visibility
        for i, trace in enumerate(fig.data):
            if trace.type == 'scatter':
                if 'line' in trace:
                    color_idx = i % len(theme["accent_colors"])
                    trace.line.color = theme["accent_colors"][color_idx]
            elif trace.type in ['bar', 'histogram']:
                color_idx = i % len(theme["accent_colors"])
                if hasattr(trace, 'marker'):
                    if hasattr(trace.marker, 'color') and trace.marker.color is not None:
                        # Don't override custom color logic (like green/red for profit/loss)
                        continue
                    trace.marker.color = theme["accent_colors"][color_idx]
        
        return fig
    
    @staticmethod
    def add_annotations(
        fig: go.Figure, 
        annotations: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Add custom annotations to a chart
        
        Args:
            fig: Plotly figure to modify
            annotations: List of annotation dictionaries with keys:
                - x: x-coordinate
                - y: y-coordinate
                - text: annotation text
                - color (optional): text color
                - arrowhead (optional): arrowhead style (0-8)
                - font_size (optional): font size
                
        Returns:
            Modified figure with annotations
        """
        for ann in annotations:
            fig.add_annotation(
                x=ann["x"],
                y=ann["y"],
                text=ann["text"],
                showarrow=True,
                arrowhead=ann.get("arrowhead", 2),
                arrowcolor=ann.get("color", COLORS["primary"]),
                font=dict(
                    color=ann.get("color", COLORS["primary"]),
                    size=ann.get("font_size", 12)
                )
            )
        
        return fig
    
    @staticmethod
    def add_highlight_regions(
        fig: go.Figure, 
        regions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Add highlighted regions to a chart
        
        Args:
            fig: Plotly figure to modify
            regions: List of region dictionaries with keys:
                - x0: start x-coordinate
                - x1: end x-coordinate
                - color (optional): fill color
                - opacity (optional): fill opacity
                - label (optional): label text
                
        Returns:
            Modified figure with highlight regions
        """
        for region in regions:
            # Add the highlight shape
            fig.add_shape(
                type="rect",
                x0=region["x0"],
                x1=region["x1"],
                y0=0,
                y1=1,
                yref="paper",
                fillcolor=region.get("color", COLORS["primary"]),
                opacity=region.get("opacity", 0.2),
                layer="below",
                line=dict(width=0)
            )
            
            # Add label if specified
            if "label" in region:
                fig.add_annotation(
                    x=region["x0"] + (region["x1"] - region["x0"]) / 2,
                    y=1,
                    yref="paper",
                    text=region["label"],
                    showarrow=False,
                    font=dict(
                        color=region.get("color", COLORS["primary"]),
                        size=10
                    )
                )
        
        return fig
    
    @staticmethod
    def customize_axis(
        fig: go.Figure, 
        axis: str,
        title: Optional[str] = None,
        range: Optional[Tuple[float, float]] = None,
        tick_format: Optional[str] = None,
        log_scale: bool = False,
        grid: bool = True
    ) -> go.Figure:
        """
        Customize a specific axis
        
        Args:
            fig: Plotly figure to modify
            axis: 'x' or 'y'
            title: Axis title
            range: Optional tuple of (min, max) for axis range
            tick_format: Optional tick format string
            log_scale: Whether to use log scale
            grid: Whether to show grid lines
            
        Returns:
            Modified figure with customized axis
        """
        axis_dict = {}
        
        if title is not None:
            axis_dict["title"] = title
            
        if range is not None:
            axis_dict["range"] = range
            
        if tick_format is not None:
            axis_dict["tickformat"] = tick_format
            
        if log_scale:
            axis_dict["type"] = "log"
            
        axis_dict["showgrid"] = grid
        
        if axis == 'x':
            fig.update_xaxes(**axis_dict)
        elif axis == 'y':
            fig.update_yaxes(**axis_dict)
            
        return fig


class AdvancedCharts:
    """Advanced chart types for financial data"""
    
    @staticmethod
    def create_candlestick_chart(
        df: pd.DataFrame,
        title: str = "Price Chart",
        volume: bool = True,
        indicators: Optional[Dict[str, pd.Series]] = None,
        theme: str = "default"
    ) -> go.Figure:
        """
        Create a candlestick chart with optional volume and indicators
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            title: Chart title
            volume: Whether to show volume
            indicators: Optional dict of indicator names and series
            theme: Chart theme
            
        Returns:
            Plotly figure with candlestick chart
        """
        # Determine subplot configuration based on what to include
        row_heights = [0.7]
        specs = [[{"secondary_y": False}]]
        subplot_titles = [title]
        
        if volume:
            row_heights.append(0.15)
            specs.append([{"secondary_y": False}])
            subplot_titles.append("Volume")
            
        if indicators and len(indicators) > 0:
            for name in indicators.keys():
                row_heights.append(0.15)
                specs.append([{"secondary_y": False}])
                subplot_titles.append(name)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=len(row_heights),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            specs=specs,
            subplot_titles=subplot_titles
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add volume trace if requested
        if volume and 'volume' in df.columns:
            colors = np.where(df['close'] >= df['open'], COLORS['success'], COLORS['danger'])
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name="Volume",
                    marker=dict(
                        color=colors,
                        opacity=0.8
                    )
                ),
                row=2, col=1
            )
        
        # Add indicator traces
        if indicators:
            row = 3 if volume else 2
            for name, series in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        name=name,
                        line=dict(width=1.5)
                    ),
                    row=row, col=1
                )
                row += 1
        
        # Apply theme
        fig = ChartCustomizer.apply_theme(fig, theme)
        
        # Update layout
        fig.update_layout(
            height=200 * len(row_heights),
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        
        return fig
    
    @staticmethod
    def create_correlation_matrix(
        df: pd.DataFrame,
        theme: str = "default"
    ) -> go.Figure:
        """
        Create a correlation matrix heatmap
        
        Args:
            df: DataFrame with columns to correlate
            theme: Chart theme
            
        Returns:
            Plotly figure with correlation matrix
        """
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )
        
        # Apply theme
        fig = ChartCustomizer.apply_theme(fig, theme)
        
        # Update layout
        fig.update_layout(
            title="Correlation Matrix",
            height=500,
            width=700,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        
        return fig


class CustomizableChart:
    """A wrapper for creating customizable charts with a UI"""
    
    def __init__(self, chart_type: str, data: Any):
        """
        Initialize the customizable chart
        
        Args:
            chart_type: Type of chart to create
            data: Data for the chart
        """
        self.chart_type = chart_type
        self.data = data
        self.title = "Chart"
        self.theme = "default"
        self.x_label = "X Axis"
        self.y_label = "Y Axis"
        self.color_by = None
        self.annotations = []
        self.highlight_regions = []
        self.log_x = False
        self.log_y = False
        
    def render_ui(self):
        """Render UI controls for customizing the chart"""
        st.markdown("### Chart Customization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.title = st.text_input("Title", value=self.title)
            self.theme = st.selectbox(
                "Theme", 
                options=list(CHART_THEMES.keys()),
                index=list(CHART_THEMES.keys()).index(self.theme)
            )
            
        with col2:
            self.x_label = st.text_input("X-Axis Label", value=self.x_label)
            self.y_label = st.text_input("Y-Axis Label", value=self.y_label)
            
        with col3:
            if isinstance(self.data, pd.DataFrame):
                columns = list(self.data.columns)
                self.color_by = st.selectbox("Color by", options=["None"] + columns)
                if self.color_by == "None":
                    self.color_by = None
                    
            self.log_x = st.checkbox("Log X-Axis", value=self.log_x)
            self.log_y = st.checkbox("Log Y-Axis", value=self.log_y)
    
    def create_chart(self) -> go.Figure:
        """Create the chart based on current settings"""
        # Convert dataframe to format needed for plotting
        if isinstance(self.data, pd.DataFrame):
            data = self.data.copy()
        else:
            data = self.data
            
        # Create basic chart based on type
        if self.chart_type == "line":
            fig = px.line(data, title=self.title, color=self.color_by)
        elif self.chart_type == "bar":
            fig = px.bar(data, title=self.title, color=self.color_by)
        elif self.chart_type == "scatter":
            fig = px.scatter(data, title=self.title, color=self.color_by)
        elif self.chart_type == "area":
            fig = px.area(data, title=self.title, color=self.color_by)
        elif self.chart_type == "pie":
            fig = px.pie(data, title=self.title)
        else:
            # Default to line chart
            fig = px.line(data, title=self.title, color=self.color_by)
            
        # Apply customizations
        fig = ChartCustomizer.apply_theme(fig, self.theme)
        
        # Customize axes
        fig = ChartCustomizer.customize_axis(
            fig, 'x', title=self.x_label, log_scale=self.log_x
        )
        fig = ChartCustomizer.customize_axis(
            fig, 'y', title=self.y_label, log_scale=self.log_y
        )
        
        # Add annotations if any
        if self.annotations:
            fig = ChartCustomizer.add_annotations(fig, self.annotations)
            
        # Add highlight regions if any
        if self.highlight_regions:
            fig = ChartCustomizer.add_highlight_regions(fig, self.highlight_regions)
            
        return fig
    
    def display(self):
        """Display the customization UI and chart"""
        self.render_ui()
        fig = self.create_chart()
        st.plotly_chart(fig, use_container_width=True)
