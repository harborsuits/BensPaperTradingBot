"""
Main Streamlit Dashboard Application for Trading Bot

This file serves as the entry point for the Streamlit dashboard, organizing
the layout and integrating all dashboard components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import plotly.figure_factory as ff
import time
from datetime import datetime, timedelta
import os
import sys
import json
from PIL import Image

# Import RealDataAdapter for MongoDB integration
try:
    from trading_bot.dashboard.services.real_data_adapter import RealDataAdapter
    has_real_adapter = True
except ImportError:
    has_real_adapter = False

# Add multiple paths to ensure modules can be found
# First, add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# Add alternate paths that might be needed in Docker
sys.path.append('/trading_bot')
sys.path.append('/')

# Create stub for missing modules if in demo/mock mode
try:
    import trading_bot.strategies.base
except ImportError:
    print("Creating mock modules for demo mode")
    # Create the missing module structure
    from types import ModuleType
    
    # Create mock modules
    if 'trading_bot.strategies' not in sys.modules:
        sys.modules['trading_bot.strategies'] = ModuleType('trading_bot.strategies')
    if 'trading_bot.strategies.base' not in sys.modules:
        sys.modules['trading_bot.strategies.base'] = ModuleType('trading_bot.strategies.base')

# Import dashboard components
from trading_bot.dashboard.components.performance_dashboard import render_performance_dashboard
from trading_bot.dashboard.components.trade_log import render_trade_log
from trading_bot.dashboard.components.active_strategies import render_active_strategies
from trading_bot.dashboard.components.alerts import render_alerts_panel
from trading_bot.dashboard.components.manual_override import render_manual_override
from trading_bot.dashboard.components.broker_balances import render_broker_balances
from trading_bot.dashboard.components.broker_accounts import render_broker_accounts, render_connection_manager
from trading_bot.dashboard.components.market_context import render_market_context
from trading_bot.dashboard.components.webhook_monitor import render_webhook_monitor
from trading_bot.dashboard.components.broker_intelligence_streamlit import render_broker_intelligence
from trading_bot.dashboard.components.broker_historical_panel import render_broker_historical_panel
from trading_bot.dashboard.components.broker_ml_predictions_panel import render_broker_ml_predictions_panel
from trading_bot.dashboard.components.risk_control_panel import create_risk_control_panel

# Import service connections
from trading_bot.dashboard.services.data_service import DataService
from trading_bot.dashboard.services.data_service_historical import extend_data_service_with_historical
from trading_bot.dashboard.services.data_service_ml import extend_data_service_with_ml
from trading_bot.dashboard.services.api_service import APIService

# Page configuration
st.set_page_config(
    page_title="BensBot Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'account_type' not in st.session_state:
    st.session_state.account_type = 'paper'  # Default to paper trading

if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'  # Default to Dashboard page

# Track if we've added the real data adapter
if 'has_real_adapter_added' not in st.session_state:
    st.session_state.has_real_adapter_added = False

# Initialize services (connection to trading bot backend)
@st.cache_resource
def init_services():
    # Create mock data service for dashboard
    class MockBaseDataService:
        def __init__(self, api_base_url=None, **kwargs):
            self.api_base_url = api_base_url
            self.cache = {}
            self.cache_expiry = {}
            self.default_cache_duration = 5
            self.is_connected = False
            self.using_real_data = False
        
        def _get_from_api(self, endpoint):
            return {}
    
    # Create data service instance with historical and ML extensions
    HistoricalDataService = extend_data_service_with_historical(MockBaseDataService)
    MLDataService = extend_data_service_with_ml(HistoricalDataService)
    
    # Initialize with appropriate parameters - all extension classes now use *kwargs
    data_service = MLDataService(
        api_base_url=None,
        historical_data_path="data/broker_performance",
        use_demo_data=True,  # Use demo data for historical performance and ML
        use_mock_data=True   # Use mock data instead of API
    )
    api_service = APIService()
    return data_service, api_service

data_service, api_service = init_services()

# Add RealDataAdapter to data service if available
if has_real_adapter and not st.session_state.has_real_adapter_added:
    try:
        # Create and attach real data adapter to the data service
        real_adapter = RealDataAdapter(data_service)
        
        # Add all methods from RealDataAdapter to data_service
        data_service.get_portfolio_summary = real_adapter.get_portfolio_summary
        data_service.get_positions = real_adapter.get_positions
        data_service.get_orders = real_adapter.get_orders
        data_service.get_trades = real_adapter.get_trades
        data_service.get_system_status = real_adapter.get_system_status
        data_service.get_market_context = real_adapter.get_market_context
        data_service.get_active_strategies = real_adapter.get_active_strategies
        data_service.get_performance_metrics = real_adapter.get_performance_metrics
        data_service.get_equity_curve = real_adapter.get_equity_curve
        
        # Mark as connected if MongoDB is available
        data_service.is_connected = real_adapter.connected
        data_service.using_real_data = real_adapter.connected
        
        # Save adapter reference to prevent garbage collection
        data_service._real_adapter = real_adapter
        
        # Mark as added
        st.session_state.has_real_adapter_added = True
        print("RealDataAdapter successfully attached to data service")
    except Exception as e:
        print(f"Error attaching RealDataAdapter: {e}")

# Define the account type here at the very top, before any sidebar or dashboard components
# This ensures it's available to all code sections below
account_type = "Live"  # Default value

# Clean, consolidated CSS for styling with maximum contrast (opposite colors)
st.markdown("""
<style>
    /* ===== 1. BASE DASHBOARD THEME ===== */
    /* Light gray background with dark text for maximum contrast */
    .stApp, .reportview-container {
        background-color: #f5f5f5;
        color: #111;
    }
    .main .block-container {
        padding: 2rem 0;
    }
    /* Clean typography with proper contrast */
    h1, h2, h3, h4 { color: black !important; font-weight: 600; }
    p, .stMarkdown, .stText, .stTextArea, .stTextInput { color: black !important; }
    
    /* ===== 2. COMPONENT STYLING ===== */
    /* Metrics: White cards with black text */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 10px 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        border: 1px solid #ddd;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: black !important;
    }
    
    /* Value indicators with clean contrast */
    .positive { color: #003300 !important; font-weight: bold; background-color: rgba(76,175,80,0.2); padding: 4px 8px; border-radius: 4px; }
    .negative { color: #660000 !important; font-weight: bold; background-color: rgba(244,67,54,0.2); padding: 4px 8px; border-radius: 4px; }
    .warning { color: #663300 !important; font-weight: bold; background-color: rgba(255,152,0,0.2); padding: 4px 8px; border-radius: 4px; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e0e0e0;
        border-radius: 4px 4px 0 0;
        padding: 0 16px;
        color: black;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* Cards and containers */
    .metric-card, .strategy-card {
        background-color: white;
        padding: 10px 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    .strategy-card { border-left: 4px solid #4CAF50; }
    .metric-card h3, .metric-card p, .metric-card span,
    .strategy-card h3, .strategy-card p, .strategy-card div { color: black !important; }
    
    /* ===== 3. SIDEBAR STYLING ===== */
    /* Dark sidebar with white text */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #222;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div, [data-testid="stSidebarNav"] span[class^="st-"],
    [data-testid="stSidebarNav"] li div a {
        color: white !important;
    }
    [data-testid="stSidebarNav"] svg { fill: white; }
    
    /* ===== 4. ACTION ELEMENTS ===== */
    /* Blue buttons with white text */
    .action-button, .stButton>button {
        background-color: #0066cc !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with system status and navigation
with st.sidebar:
    st.title("BensBot Trading System")
    
    # Make sure sidebar text is white on dark background
    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # System Status
    st.subheader("System Status")
    try:
        system_status = data_service.get_system_status()
    except AttributeError:
        # Create a default system status if the method doesn't exist
        system_status = {
            "trading_enabled": False,
            "status": "offline",
            "uptime": "N/A",
            "last_update": "N/A",
            "message": "Running in demo mode"
        }
    
    col1, col2 = st.columns(2)
    with col1:
        if system_status["trading_enabled"]:
            st.success("Trading: Enabled")
        else:
            st.error("Trading: Disabled")
    with col2:
        uptime = system_status.get("uptime", "Unknown")
        st.info(f"Uptime: {uptime}")
    
    # Global stats
    # Get portfolio summary data with safe fallback for account type parameter
    try:
        try:
            portfolio_summary = data_service.get_portfolio_summary(account_type=st.session_state.account_type)
        except AttributeError:
            # Create default portfolio summary if method doesn't exist
            portfolio_summary = {
                "cash_balance": 100000.0,
                "total_equity": 100000.0,
                "securities_value": 0.0,
                "open_positions": 0,
                "buying_power": 100000.0,
                "margin_used": 0.0,
                "margin_available": 100000.0,
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0
            }
    except TypeError:
        # Fallback if method doesn't accept account_type parameter
        try:
            portfolio_summary = data_service.get_portfolio_summary()
        except AttributeError:
            # Create default portfolio summary if method doesn't exist at all
            portfolio_summary = {
                "cash_balance": 100000.0,
                "total_equity": 100000.0,
                "securities_value": 0.0,
                "open_positions": 0,
                "buying_power": 100000.0,
                "margin_used": 0.0,
                "margin_available": 100000.0,
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0
            }
    total_equity = portfolio_summary.get('total_equity', 0)
    daily_pnl = portfolio_summary.get('daily_pnl', 0)
    daily_pnl_pct = portfolio_summary.get('daily_pnl_pct', 0)
    
    st.subheader("Portfolio Summary")
    
    # Add styling to make sidebar portfolio numbers black on white background with more specific selectors
    st.markdown("""
    <style>
    /* Override sidebar metric styling with white cards and black text */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: white !important;
        padding: 10px !important;
        border-radius: 5px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        margin-bottom: 1rem !important;
    }
    /* Force ALL text elements in sidebar metrics to be black */
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricDelta"],
    [data-testid="stSidebar"] [data-testid="stMetric"] div,
    [data-testid="stSidebar"] [data-testid="stMetric"] p,
    [data-testid="stSidebar"] [data-testid="stMetric"] span,
    [data-testid="stSidebar"] [data-testid="stMetric"] label,
    [data-testid="stSidebar"] [data-testid="stMetric"] * {
        color: black !important;
        font-weight: bold !important;
    }
    /* Ensure background of delta is white */
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.metric(
        "Total Portfolio Value", 
        f"${total_equity:,.2f}",
        f"{daily_pnl_pct:.2f}% Today"
    )
    
    # Navigation
    # Create tabs for main dashboard components
    tabs = st.tabs([
        "Performance", 
        "Broker Accounts",
        "Active Strategies", 
        "Trade Log", 
        "News & Market Context",
        "Manual Control",
        "Risk Management",
        "System Monitor"
    ])
    
    # Manual Override Panel (Always visible)
    st.subheader("Emergency Controls")
    try:
        render_manual_override(api_service, account_type=st.session_state.account_type if 'account_type' in st.session_state else 'Live')
    except TypeError:
        # Fallback if method doesn't accept account_type parameter
        render_manual_override(api_service)

# Main content area
st.title("Trading Dashboard")

# Account toggle styling and configuration is already defined above

# Make sure we apply consistent styling for all metrics to ensure full number visibility
st.markdown("""
<style>
/* Ensure metric values are fully visible and right-aligned */
[data-testid="stMetricValue"] {
    display: flex !important;
    justify-content: flex-end !important;
    flex-grow: 1 !important;
    font-family: monospace !important;
    font-size: 1.1rem !important;
    font-variant-numeric: tabular-nums !important;
}
/* Make metrics cards expand to full width */
[data-testid="stMetric"] {
    width: 100% !important;
    margin-bottom: 10px !important;
    padding: 15px !important;
    box-sizing: border-box !important;
}
/* Center all headers */
h1, h2, h3, h4, h5, h6 {
    text-align: center !important;
}
/* Fix button alignment */
.stButton {
    display: flex !important;
    justify-content: center !important;
}
</style>
""", unsafe_allow_html=True)

# Define default page if not defined
page = st.session_state.get('page', 'Dashboard')

if page == "Dashboard":
    # Portfolio section with clear account type toggle
    st.subheader("Portfolio Overview", anchor=False)
    
    # Create a prominent account toggle in its own container
    # Commenting out the invalid rerun call until proper container context is established
    # st.rerun()
    
    # System status indicators
    st.markdown(
        f"""<div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
            <span style="background-color: {'#4CAF50' if system_status.get('status') == 'running' else '#F44336'}; color: white; padding: 5px 10px; border-radius: 4px; margin-right: 10px;">
                {system_status.get('status', 'offline').upper()}
            </span>
            <span style="color: #666; font-size: 0.9rem;">{system_status.get('last_update', 'Unknown')} | v{system_status.get('version', '1.0.0')}</span>
        </div>""",
        unsafe_allow_html=True
    )
    # Add special styling for the main section metrics to ensure black text on white background
    st.markdown("""
    <style>
    /* Override specific metrics for the main section to ensure black text on white background */
    .main [data-testid=\"stMetricLabel\"] { 
        color: black !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-align: center !important;
    }
    .main [data-testid=\"stMetricValue\"] {
        color: black !important;
        font-weight: 700 !important;
        min-width: 100% !important;
        text-align: center !important;
        font-variant-numeric: tabular-nums !important;
        font-family: monospace !important;
        font-size: 1.5rem !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    .main [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        text-align: center !important;
    }
    /* Make metrics display as symmetrical cards */
    [data-testid="stMetric"] {
        background-color: white !important;
        border: 1px solid #ddd !important;
        border-radius: 5px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
        height: 130px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        width: 100% !important;
    }
    /* Fix vertical stacking issues */
    [data-testid="column"] { 
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create vertical stacked metrics
    # 1. Total P&L
    total_pnl = portfolio_summary.get('total_pnl', 0)
    pnl_change = portfolio_summary.get('total_pnl_pct', 0)
    st.metric(
        "Total P&L", 
        f"${total_pnl:,.2f}", 
        f"{pnl_change:.2f}%", 
        delta_color="normal" if pnl_change >= 0 else "inverse"
    )
    
    # 2. Win Rate
    win_rate = portfolio_summary.get('win_rate', 0)
    win_rate_change = portfolio_summary.get('win_rate_change', 0)
    st.metric(
        "Win Rate", 
        f"{win_rate:.2f}%", 
        f"{win_rate_change:+.2f}%", 
        delta_color="normal" if win_rate_change >= 0 else "inverse"
    )
    
    # 3. Trade Count
    trades_today = portfolio_summary.get('trades_today', 0)
    trades_change = portfolio_summary.get('trades_change', 0)
    st.metric(
        "Trades (Today)", 
        f"{trades_today:,d}", 
        f"{trades_change:+d}", 
        delta_color="normal" if trades_change >= 0 else "inverse"
    )
    
    # 4. Market Regime
    try:
        market_data = data_service.get_market_context()
    except AttributeError:
        # Create mock market data if method doesn't exist
        market_data = {
            "market_conditions": "NORMAL",
            "volatility": "MEDIUM",
            "trend": "NEUTRAL",
            "risk_level": "MODERATE",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    regime = market_data.get('market_regime', 'Neutral')
    
    # Create a styled metric for Market Regime that matches the other metrics
    st.metric(
        "Market Regime",
        regime,
        None
    )
    
    # Main dashboard layout in two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Active Strategies Monitor - pass account type filter from session state
        st.subheader("Active Strategies")
        try:
            render_active_strategies(data_service, api_service, st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_active_strategies(data_service, api_service)
        
        # Performance Dashboard (Simplified version for main page)
        st.subheader("Strategy Performance Snapshot")
        try:
            render_performance_dashboard(data_service, simplified=True, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_performance_dashboard(data_service, simplified=True)
    
    with col2:
        # Alerts Panel
        st.subheader("Recent Alerts")
        try:
            render_alerts_panel(data_service, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_alerts_panel(data_service)
        
        # Trade Log (Recent only)
        st.subheader("Recent Trades")
        try:
            render_trade_log(data_service, max_trades=10, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_trade_log(data_service, max_trades=10)
        
        # Broker Balances (Simplified) - filtered by account type from session state
        st.subheader("Broker Accounts")
        try:
            render_broker_balances(data_service, simplified=True, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_broker_balances(data_service, simplified=True)

elif page == "Strategy Performance":
    st.header("Strategy Performance Analysis")
    
    # Time period selector
    time_period = st.selectbox(
        "Select Time Period",
        ["Today", "Yesterday", "This Week", "This Month", "All Time"]
    )
    
    # Detailed performance dashboard
    try:
        render_performance_dashboard(data_service, time_period=time_period, account_type=st.session_state.account_type)
    except TypeError:
        # Fallback if method doesn't accept account_type parameter
        render_performance_dashboard(data_service, time_period=time_period)

elif page == "Trade Log":
    st.header("Trade Log and Signal Monitor")
    
    # Tabs for different log types
    tab1, tab2 = st.tabs(["Trade Log", "Webhook Signals"])
    
    with tab1:
        # Full trade log with filtering
        try:
            render_trade_log(data_service, with_filters=True, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_trade_log(data_service, with_filters=True)
    
    with tab2:
        # Webhook signal monitor
        try:
            render_webhook_monitor(data_service, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_webhook_monitor(data_service)

elif page == "Broker & Positions":
    st.header("Broker Accounts & Positions")
    
    # Create tabs for broker information
    broker_tabs = st.tabs(["Accounts & Balances", "Positions", "Metrics", "Historical Performance", "ML Predictions", "Broker Intelligence"])
    
    with broker_tabs[0]:
        # Full broker balances and positions view
        try:
            render_broker_balances(data_service, account_type=st.session_state.account_type)
        except TypeError:
            # Fallback if method doesn't accept account_type parameter
            render_broker_balances(data_service)
    
    with broker_tabs[1]:
        # Positions panel
        st.subheader("Positions")
        # render_broker_positions_panel(data_service)
    
    with broker_tabs[2]:
        # Metrics panel
        st.subheader("Metrics")
        # render_broker_metrics_panel(data_service)
    
    with broker_tabs[3]:
        # Historical performance panel
        st.subheader("Historical Performance")
        render_broker_historical_panel(data_service)
    
    with broker_tabs[4]:
        # ML predictions panel
        st.subheader("ML Predictions")
        try:
            render_broker_ml_predictions_panel(data_service)
        except Exception as e:
            st.error(f"Error rendering ML predictions: {str(e)}")
    
    with broker_tabs[5]:
        # Broker intelligence dashboard
        try:
            render_broker_intelligence(data_service, account_type=st.session_state.account_type)
        except Exception as e:
            st.error(f"Error rendering broker intelligence: {str(e)}")

elif page == "Market Context":
    st.header("Market Context & Sentiment")
    
    # Market context and sentiment panel
    try:
        render_market_context(data_service, account_type=st.session_state.account_type)
    except TypeError:
        # Fallback if method doesn't accept account_type parameter
        render_market_context(data_service)

elif page == "Settings":
    st.header("System Settings")
    
    # Settings panel (placeholder)
    st.write("Settings panel under development")

elif page == "Risk Management":
    st.header("Risk Management & Circuit Breakers")
    
    # Create risk control panel
    risk_panel = create_risk_control_panel(data_service)
    risk_panel.render()

# Auto-refresh mechanism
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto_refresh:
    st.empty()
    time.sleep(5)
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.caption("BensBot Trading System © 2025")
