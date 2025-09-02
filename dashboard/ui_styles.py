"""
UI Styles Module

This module defines the visual language for the trading dashboard, including:
- Color schemes (dark/light themes)
- Typography settings
- Component styling
- Animation properties
"""

import streamlit as st
from enum import Enum

class ThemeMode(Enum):
    DARK = "dark"
    LIGHT = "light"

class UIColors:
    """Color palette definitions for the application"""
    
    # Primary theme colors
    class Dark:
        # Background colors
        BG_PRIMARY = "#121212"
        BG_SECONDARY = "#1E1E1E"
        BG_TERTIARY = "#2D2D2D"
        BG_CARD = "#252525"
        
        # Text colors
        TEXT_PRIMARY = "#FFFFFF"
        TEXT_SECONDARY = "#B0B0B0"
        TEXT_TERTIARY = "#787878"
        
        # Accent colors
        ACCENT_PRIMARY = "#4F8BFF"  # Blue
        ACCENT_SECONDARY = "#9C67FF"  # Purple
        
        # Status colors
        SUCCESS = "#4CAF50"  # Green
        WARNING = "#FF9800"  # Orange
        ERROR = "#F44336"  # Red
        INFO = "#2196F3"  # Light Blue
        
        # Trading specific
        PROFIT = "#00C853"  # Bright Green
        LOSS = "#FF5252"  # Bright Red
        NEUTRAL = "#9E9E9E"  # Gray
        
        # Chart colors
        CHART_GRID = "#333333"
        CHART_LINE = "#666666"
        CHART_ACCENT = "#4F8BFF"
        
        # Gradient stops
        GRADIENT_START = "#2C3E50"
        GRADIENT_END = "#4CA1AF"
    
    class Light:
        # Background colors
        BG_PRIMARY = "#FFFFFF"
        BG_SECONDARY = "#F5F5F5"
        BG_TERTIARY = "#EEEEEE"
        BG_CARD = "#FFFFFF"
        
        # Text colors
        TEXT_PRIMARY = "#212121"
        TEXT_SECONDARY = "#757575"
        TEXT_TERTIARY = "#9E9E9E"
        
        # Accent colors
        ACCENT_PRIMARY = "#1565C0"  # Darker Blue
        ACCENT_SECONDARY = "#6A1B9A"  # Darker Purple
        
        # Status colors
        SUCCESS = "#2E7D32"  # Darker Green
        WARNING = "#EF6C00"  # Darker Orange
        ERROR = "#C62828"  # Darker Red
        INFO = "#0277BD"  # Darker Light Blue
        
        # Trading specific
        PROFIT = "#00C853"  # Bright Green
        LOSS = "#D32F2F"  # Darker Red
        NEUTRAL = "#757575"  # Darker Gray
        
        # Chart colors
        CHART_GRID = "#E0E0E0"
        CHART_LINE = "#BDBDBD"
        CHART_ACCENT = "#1976D2"
        
        # Gradient stops
        GRADIENT_START = "#E0EAFC"
        GRADIENT_END = "#CFDEF3"

class UITypography:
    """Typography settings for the application"""
    
    # Font families
    FONT_PRIMARY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    FONT_SECONDARY = "Roboto, Arial, sans-serif"
    FONT_MONO = "'Roboto Mono', 'Courier New', monospace"
    
    # Font sizes
    H1 = "2.125rem"  # 34px
    H2 = "1.875rem"  # 30px
    H3 = "1.5rem"    # 24px
    H4 = "1.25rem"   # 20px
    H5 = "1rem"      # 16px
    BODY_LARGE = "1rem"      # 16px
    BODY = "0.875rem"        # 14px
    BODY_SMALL = "0.8125rem" # 13px
    CAPTION = "0.75rem"      # 12px
    
    # Font weights
    WEIGHT_LIGHT = 300
    WEIGHT_REGULAR = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700
    
    # Line heights
    LINE_HEIGHT_TIGHT = 1.2
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.8
    
    # Letter spacing
    LETTER_SPACING_TIGHT = "-0.01em"
    LETTER_SPACING_NORMAL = "0"
    LETTER_SPACING_WIDE = "0.01em"

class UIEffects:
    """Animation and effect settings"""
    
    # Shadows
    SHADOW_NONE = "none"
    SHADOW_XS = "0 1px 2px rgba(0, 0, 0, 0.05)"
    SHADOW_SM = "0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)"
    SHADOW_MD = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    SHADOW_LG = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    SHADOW_XL = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
    
    # Border radius
    RADIUS_NONE = "0"
    RADIUS_XS = "2px"
    RADIUS_SM = "4px"
    RADIUS_MD = "6px"
    RADIUS_LG = "8px"
    RADIUS_XL = "12px"
    RADIUS_FULL = "9999px"
    
    # Transitions
    TRANSITION_FAST = "all 0.15s ease"
    TRANSITION_NORMAL = "all 0.25s ease"
    TRANSITION_SLOW = "all 0.4s ease"

class UISpacing:
    """Spacing system for consistent margins and padding"""
    
    XS = "0.25rem"  # 4px
    SM = "0.5rem"   # 8px
    MD = "1rem"     # 16px
    LG = "1.5rem"   # 24px
    XL = "2rem"     # 32px
    XXL = "3rem"    # 48px

def apply_base_styles(theme_mode=ThemeMode.DARK):
    """Apply base styles to the Streamlit application"""
    
    colors = UIColors.Dark if theme_mode == ThemeMode.DARK else UIColors.Light
    
    # Define CSS
    st.markdown(f"""
    <style>
        /* Base styles */
        [data-testid="stAppViewContainer"] {{
            background-color: {colors.BG_PRIMARY};
            color: {colors.TEXT_PRIMARY};
            font-family: {UITypography.FONT_PRIMARY};
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {colors.BG_SECONDARY};
            padding: {UISpacing.MD};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            font-family: {UITypography.FONT_PRIMARY};
            font-weight: {UITypography.WEIGHT_BOLD};
            color: {colors.TEXT_PRIMARY};
            margin-bottom: {UISpacing.MD};
        }}
        
        h1 {{
            font-size: {UITypography.H1};
            letter-spacing: {UITypography.LETTER_SPACING_TIGHT};
        }}
        
        h2 {{
            font-size: {UITypography.H2};
            letter-spacing: {UITypography.LETTER_SPACING_TIGHT};
        }}
        
        h3 {{
            font-size: {UITypography.H3};
        }}
        
        /* Text elements */
        p {{
            font-size: {UITypography.BODY};
            line-height: {UITypography.LINE_HEIGHT_NORMAL};
            color: {colors.TEXT_SECONDARY};
        }}
        
        /* Card styling */
        .card {{
            background-color: {colors.BG_CARD};
            border-radius: {UIEffects.RADIUS_MD};
            padding: {UISpacing.MD};
            box-shadow: {UIEffects.SHADOW_MD};
            margin-bottom: {UISpacing.MD};
            transition: {UIEffects.TRANSITION_NORMAL};
        }}
        
        .card:hover {{
            box-shadow: {UIEffects.SHADOW_LG};
        }}
        
        /* Button styling overrides */
        .stButton > button {{
            background-color: {colors.ACCENT_PRIMARY};
            color: white;
            border: none;
            border-radius: {UIEffects.RADIUS_MD};
            padding: {UISpacing.SM} {UISpacing.MD};
            font-weight: {UITypography.WEIGHT_MEDIUM};
            transition: {UIEffects.TRANSITION_FAST};
        }}
        
        .stButton > button:hover {{
            background-color: {colors.ACCENT_SECONDARY};
            box-shadow: {UIEffects.SHADOW_MD};
        }}
        
        /* Metrics styling */
        .metric-card {{
            background-color: {colors.BG_CARD};
            border-radius: {UIEffects.RADIUS_MD};
            padding: {UISpacing.MD};
            border-left: 4px solid {colors.ACCENT_PRIMARY};
            margin-bottom: {UISpacing.MD};
        }}
        
        .metric-value {{
            font-size: {UITypography.H3};
            font-weight: {UITypography.WEIGHT_BOLD};
            margin-bottom: 0;
        }}
        
        .metric-label {{
            font-size: {UITypography.CAPTION};
            color: {colors.TEXT_TERTIARY};
            margin-top: 0;
        }}
        
        /* Status indicators */
        .status-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        
        .status-dot.success {{
            background-color: {colors.SUCCESS};
        }}
        
        .status-dot.warning {{
            background-color: {colors.WARNING};
        }}
        
        .status-dot.error {{
            background-color: {colors.ERROR};
        }}
        
        .status-dot.neutral {{
            background-color: {colors.NEUTRAL};
        }}
        
        /* Trading specific */
        .profit {{
            color: {colors.PROFIT} !important;
        }}
        
        .loss {{
            color: {colors.LOSS} !important;
        }}
        
        /* Form elements */
        div[data-baseweb="select"] {{
            background-color: {colors.BG_TERTIARY};
            border-radius: {UIEffects.RADIUS_MD};
        }}
        
        /* Data display tables */
        [data-testid="stTable"] {{
            background-color: {colors.BG_CARD};
            border-radius: {UIEffects.RADIUS_MD};
            overflow: hidden;
        }}
        
        /* Animation for loading states */
        @keyframes pulse {{
            0% {{ opacity: 0.6; }}
            50% {{ opacity: 0.8; }}
            100% {{ opacity: 0.6; }}
        }}
        
        .loading {{
            animation: pulse 1.5s infinite;
            background-color: {colors.BG_TERTIARY};
            border-radius: {UIEffects.RADIUS_MD};
        }}
        
        /* Chart customizations */
        .chart-container {{
            background-color: {colors.BG_CARD};
            border-radius: {UIEffects.RADIUS_MD};
            padding: {UISpacing.MD};
            box-shadow: {UIEffects.SHADOW_SM};
        }}
    </style>
    """, unsafe_allow_html=True)

def format_currency(value, precision=2):
    """Format value as currency"""
    if value >= 0:
        return f"<span class='profit'>${value:,.{precision}f}</span>"
    else:
        return f"<span class='loss'>${abs(value):,.{precision}f}</span>"

def format_percentage(value, precision=2):
    """Format value as percentage"""
    if value >= 0:
        return f"<span class='profit'>+{value:.{precision}f}%</span>"
    else:
        return f"<span class='loss'>{value:.{precision}f}%</span>"

def create_card(title, content, icon=None, status=None):
    """Create a styled card component"""
    status_html = ""
    if status:
        status_class = status.lower()
        status_html = f'<span class="status-dot {status_class}"></span>'
    
    icon_html = f'<i class="fas {icon} mr-2"></i>' if icon else ''
    
    return f"""
    <div class="card">
        <h3>{icon_html}{status_html}{title}</h3>
        <div>{content}</div>
    </div>
    """

def create_metric_card(label, value, previous_value=None, show_delta=True, precision=2):
    """Create a metric card with optional delta indicator"""
    delta_html = ""
    
    if previous_value is not None and show_delta:
        delta = value - previous_value
        delta_percentage = (delta / abs(previous_value)) * 100 if previous_value != 0 else 0
        delta_class = 'profit' if delta >= 0 else 'loss'
        delta_symbol = '+' if delta >= 0 else ''
        delta_html = f'<span class="{delta_class}">({delta_symbol}{delta_percentage:.{precision}f}%)</span>'
    
    formatted_value = format_currency(value, precision) if isinstance(value, (int, float)) else value
    
    return f"""
    <div class="metric-card">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{formatted_value} {delta_html}</p>
    </div>
    """

def theme_plotly_chart(fig, theme_mode=ThemeMode.DARK):
    """Apply theme to Plotly chart"""
    colors = UIColors.Dark if theme_mode == ThemeMode.DARK else UIColors.Light
    
    fig.update_layout(
        plot_bgcolor=colors.BG_CARD,
        paper_bgcolor=colors.BG_CARD,
        font_color=colors.TEXT_PRIMARY,
        font_family=UITypography.FONT_PRIMARY,
        title_font_family=UITypography.FONT_PRIMARY,
        legend_font_family=UITypography.FONT_PRIMARY,
        colorway=[colors.ACCENT_PRIMARY, colors.ACCENT_SECONDARY, colors.SUCCESS, colors.WARNING, 
                  colors.ERROR, colors.INFO, "#FF6F91", "#845EC2", "#F9F871", "#00C9A7"],
        margin=dict(l=10, r=10, t=30, b=10),
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors.CHART_GRID,
        showline=True,
        linewidth=1,
        linecolor=colors.CHART_LINE,
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors.CHART_GRID,
        showline=True,
        linewidth=1,
        linecolor=colors.CHART_LINE,
    )
    
    return fig
