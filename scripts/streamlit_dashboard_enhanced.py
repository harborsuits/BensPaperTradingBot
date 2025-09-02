"""
BensBot Trading Dashboard

An autonomous trading dashboard built with Streamlit that integrates with
the existing event-driven trading architecture.
"""
import os
import json
import datetime
import asyncio
import websockets
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
import logging

# Internal imports from the dashboard package
from dashboard.theme import apply_custom_styling, COLORS
from dashboard.api_utils import (
    get_portfolio_data, get_trades, get_strategies, 
    get_alerts, get_system_logs, get_event_bus_status,
    get_trading_modes
)
from dashboard.visualizations import (
    df_or_empty, create_performance_chart, create_pie_chart,
    create_trade_history_chart, create_event_system_chart,
    enhance_dataframe
)
from dashboard.components import (
    section_header, styled_metric_card, strategy_card,
    strategy_lane, strategy_status_badge,
    event_system_status_card, trading_mode_card
)

# Import new modules
from dashboard.data_sources import (
    AlphaVantageAPI, FinnhubAPI, NewsDataAPI,
    _api_cache, _cache_expiry
)
from dashboard.backtesting import (
    calculate_performance_metrics, create_equity_curve_chart,
    create_monthly_returns_heatmap, create_distribution_chart,
    generate_customizable_backtest_metrics, display_backtest_metrics_dashboard,
    get_mock_backtest_data
)
from dashboard.custom_visualizations import (
    ChartCustomizer, AdvancedCharts, CustomizableChart, CHART_THEMES
)
from dashboard.advanced_monitoring import (
    SystemMonitor, LogAnalyzer, AlertManager,
    system_monitor, alert_manager
)
from dashboard.risk_management import risk_dashboard
from dashboard.strategy_rotation import strategy_rotation_dashboard
from dashboard.real_time_alerts import real_time_dashboard
from dashboard.historical_backtest import backtest_dashboard
from dashboard.telegram_integration import telegram_dashboard

# Dictionary mapping page names to their corresponding functions
PAGES = {
    "Dashboard Overview": "Dashboard",
    "Performance Monitoring": "Performance Monitoring",
    "Strategy Monitoring": "Strategies",
    "Strategy Rotation": strategy_rotation_dashboard,
    "Risk Management": risk_dashboard,
    "Real-Time Alerts": real_time_dashboard,
    "Historical Backtests": backtest_dashboard,
    "Trade Monitoring": "Trade Monitoring",
    "Position Monitoring": "Position Monitoring",
    "Event Monitoring": "Event Monitoring",
    "System Monitoring": "System Monitor"
}

# Integration with your existing trading system 
try:
    from trading_bot.event_system import EventManager, EventBus, MessageQueue, ChannelManager
    from trading_bot.trading_modes import BaseTradingMode, StandardTradingMode
    from trading_bot.order_manager import Order, OrderType
    from trading_bot.config import Config
    HAS_TRADING_BOT = True
    
    # Load config for API keys
    config = Config()
    # Set environment variables for API keys
    os.environ["ALPHA_VANTAGE_API_KEY"] = config.ALPHA_VANTAGE_API_KEY
    os.environ["FINNHUB_API_KEY"] = config.FINNHUB_API_KEY
    os.environ["NEWSDATA_API_KEY"] = config.NEWSDATA_API_KEY
    os.environ["MARKETAUX_API_KEY"] = config.MARKETAUX_API_KEY
    os.environ["TRADIER_API_KEY"] = config.TRADIER_API_KEY
    os.environ["ALPACA_API_KEY"] = config.ALPACA_API_KEY
    os.environ["ALPACA_SECRET_KEY"] = config.ALPACA_SECRET_KEY
except ImportError:
    HAS_TRADING_BOT = False
    st.warning("Some trading_bot modules could not be imported. The dashboard will run in limited mode.")
    # For development without trading_bot, set default API keys if available
    if "ALPHA_VANTAGE_API_KEY" not in os.environ:
        os.environ["ALPHA_VANTAGE_API_KEY"] = ""

# ---------------------------------------------------------------------------
# WebSocket Integration for Real-time Updates
# ---------------------------------------------------------------------------
async def connect_to_metrics_websocket():
    """Connect to the WebSocket endpoint for real-time metrics"""
    uri = "ws://localhost:8000/ws/metrics"
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                data = await websocket.recv()
                # Parse the metrics data
                metrics = json.loads(data)
                # Store the metrics in session state
                st.session_state.websocket_metrics = metrics
    except Exception as e:
        st.session_state.websocket_error = str(e)
        print(f"WebSocket error: {e}")

def initialize_websocket():
    """Initialize the WebSocket connection if not already running"""
    if "websocket_initialized" not in st.session_state:
        st.session_state.websocket_initialized = True
        st.session_state.websocket_metrics = {}
        st.session_state.websocket_error = None
        
        # Start the WebSocket connection in a background thread
        loop = asyncio.new_event_loop()
        
        def run_websocket():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(connect_to_metrics_websocket())
            
        import threading
        websocket_thread = threading.Thread(target=run_websocket, daemon=True)
        websocket_thread.start()

# ---------------------------------------------------------------------------
# Main Dashboard Application
# ---------------------------------------------------------------------------

# Configure the page settings
st.set_page_config(
    page_title="BensBot Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_styling()

# Initialize WebSocket connection
initialize_websocket()

# Create sidebar for navigation
st.sidebar.title("BensBot Dashboard")
try:
    st.sidebar.image("dashboard/assets/bensbot_logo.png", width=200)
except:
    pass  # Continue if image not found

# Organize pages into categories
st.sidebar.markdown("### Main")
selected_main_page = st.sidebar.radio(
    "",
    ["Dashboard", "Performance Monitoring"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### Strategy")
selected_strategy_page = st.sidebar.radio(
    "",
    ["Strategies", "Strategy Rotation"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### Risk Management")
selected_risk_page = st.sidebar.radio(
    "",
    ["Risk Management", "Real-Time Alerts", "Historical Backtests"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### Notifications")
selected_notifications_page = st.sidebar.radio(
    "",
    ["Telegram Alerts"],
    label_visibility="collapsed"
)

st.sidebar.markdown("### Detailed Monitoring")
selected_detailed_page = st.sidebar.radio(
    "",
    ["Trade Monitoring", "Position Monitoring", "Event Monitoring", "System Monitor"],
    label_visibility="collapsed"
)

# Determine which page to show
if selected_main_page != "Dashboard":
    page = selected_main_page
elif selected_strategy_page != "Strategies":
    page = selected_strategy_page
elif selected_risk_page != "Risk Management":
    page = selected_risk_page
elif selected_notifications_page == "Telegram Alerts":
    page = "Telegram Alerts"
elif selected_detailed_page != "Trade Monitoring":
    page = selected_detailed_page
else:
    page = "Dashboard"

# Settings section in sidebar
st.sidebar.markdown("---")
st.sidebar.title("Settings")
refresh_interval = st.sidebar.slider(
    "Refresh Interval (sec)", 
    min_value=5, 
    max_value=300, 
    value=60,
    step=5
)

# Automated refresh checkbox
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

if auto_refresh:
    st.sidebar.info(f"Dashboard will refresh every {refresh_interval} seconds")
    
# Add a manual refresh button
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# Main content area - changes based on selected page
if page == "Dashboard":
    # Main dashboard page
    st.title("Trading Dashboard")
    
    # Top metrics row
    st.subheader("Portfolio Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        styled_metric_card("Portfolio Value", "$125,432.10", delta=2.3, is_currency=True)
    
    with col2:
        styled_metric_card("Daily P&L", "$1,432.10", delta=1.2, is_currency=True)
    
    with col3:
        styled_metric_card("Win Rate", "68%", is_percent=True)
        
# Call the appropriate dashboard function based on selected page
elif page == "Strategy Rotation":
    strategy_rotation_dashboard()
elif page == "Risk Management":
    risk_dashboard()
elif page == "Real-Time Alerts":
    real_time_dashboard()
elif page == "Historical Backtests":
    backtest_dashboard()
elif page == "Telegram Alerts":
    telegram_dashboard()
    
    with col4:
        styled_metric_card("Drawdown", "3.2%", delta=-0.5, is_percent=True)
    
    # Performance chart
    section_header("Performance Chart", icon="📈")
    st.plotly_chart(create_performance_chart(), use_container_width=True)
    
    # Strategy status lanes
    section_header("Strategy Status", icon="🔢")
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_lane("active", title="Active Strategies", icon="✅", action="retire")
    
    with col2:
        strategy_lane("experimental", title="Experimental Strategies", icon="🧪", action="promote")
    
    # Bottom row with system status and trading mode
    col1, col2 = st.columns(2)
    
    with col1:
        event_system_status_card()
    
    with col2:
        trading_mode_card()

elif page == "Strategies":
    st.title("Strategy Management")
    
    tab1, tab2, tab3 = st.tabs(["Active Strategies", "Strategy Performance", "Strategy Creation"])
    
    with tab1:
        st.subheader("Active Trading Strategies")
        strategies = get_strategies()
        
        for strategy in strategies:
            strategy_card(strategy, action="view")
    
    with tab2:
        st.subheader("Strategy Performance Comparison")
        # Strategy performance visualization would go here
        st.info("Strategy performance comparison charts coming soon")
    
    with tab3:
        st.subheader("Create New Strategy")
        st.info("Strategy creation interface coming soon")

elif page == "Risk Management":
    # Integrated Risk Management Dashboard
    risk_dashboard()

elif page == "Backtesting":
    st.title("Backtesting Lab")
    
    # Get mock backtest data for demonstration
    backtest_data = get_mock_backtest_data()
    
    # Display backtest metrics dashboard
    display_backtest_metrics_dashboard(backtest_data)

elif page == "Market Data":
    st.title("Market Data & Analysis")
    
    # Tabs for different market data views
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Watchlist", "Economic Calendar"])
    
    with tab1:
        st.subheader("Market Overview")
        st.info("Market overview charts coming soon")
    
    with tab2:
        st.subheader("Watchlist")
        st.info("Symbol watchlist coming soon")
    
    with tab3:
        st.subheader("Economic Calendar")
        st.info("Economic calendar coming soon")

elif page == "System Monitor":
    st.title("System Monitoring")
    
    # Get system monitoring data
    system_status = system_monitor.get_status()
    alerts = alert_manager.get_recent_alerts()
    logs = get_system_logs()
    
    # Display system metrics
    section_header("System Health", icon="💻")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        styled_metric_card("CPU Usage", f"{system_status.get('cpu_usage', 0)}%", is_percent=True)
    
    with col2:
        styled_metric_card("Memory Usage", f"{system_status.get('memory_usage', 0)}%", is_percent=True)
    
    with col3:
        styled_metric_card("Uptime", f"{system_status.get('uptime', 0)} hours")
    
    # Display recent alerts
    section_header("Recent Alerts", icon="🔔")
    
    if alerts:
        for alert in alerts:
            st.warning(f"**{alert.get('level', 'Info')}**: {alert.get('message', '')}")
            st.text(f"Time: {alert.get('timestamp', '')}")
            st.markdown("---")
    else:
        st.success("No recent alerts - all systems running normally")
    
    # Display recent logs
    section_header("System Logs", icon="📝")
    
    if logs:
        log_df = pd.DataFrame(logs)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No recent logs available")
