#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Main UI

A comprehensive dashboard for monitoring and controlling the BensBot 
autonomous trading system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import datetime
import json
import os
import sys
from pathlib import Path
import logging

# Ensure proper path for imports
root_dir = str(Path(__file__).parent.parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import dashboard components
from trading_bot.dashboard.new_ui.components.portfolio_summary import render_portfolio_summary
from trading_bot.dashboard.new_ui.components.active_strategies import render_active_strategies
from trading_bot.dashboard.new_ui.components.trade_log import render_trade_log
from trading_bot.dashboard.new_ui.components.alert_feed import render_alert_feed
from trading_bot.dashboard.new_ui.components.strategy_lab import render_strategy_lab
from trading_bot.dashboard.new_ui.components.manual_controls import render_manual_controls
from trading_bot.dashboard.new_ui.data_service import DataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BensBot-Dashboard")

# Main application
def main():
    # Page config
    st.set_page_config(
        page_title="BensBot Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for data persistence
    if 'initialized' not in st.session_state:
        initialize_session_state()
        
    # Sidebar navigation
    st.sidebar.title("BensBot Trading Dashboard")
    st.sidebar.image("https://img.icons8.com/color/96/000000/robot.png", width=100)
    
    # Status indicator in sidebar
    if st.session_state.trading_active:
        st.sidebar.success("⚡ Trading Active")
    else:
        st.sidebar.warning("⏸️ Trading Paused")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Portfolio Overview", "Active Strategies", "Trade Log", "Alerts", "Strategy Lab"]
    )
    
    # Always show manual controls in sidebar
    with st.sidebar.expander("🎮 Manual Controls", expanded=True):
        render_manual_controls()
    
    # Main content based on navigation
    if page == "Portfolio Overview":
        render_overview_page()
    elif page == "Active Strategies":
        render_strategies_page()
    elif page == "Trade Log":
        render_trade_log_page()
    elif page == "Alerts":
        render_alerts_page()
    elif page == "Strategy Lab":
        render_strategy_lab_page()
    
    # Background update logic
    trigger_background_update()

def initialize_session_state():
    """Initialize all session state variables"""
    # System state
    st.session_state.initialized = True
    st.session_state.trading_active = False
    st.session_state.last_update_time = None
    st.session_state.broker_connected = False
    st.session_state.connection_status = "Disconnected"
    
    # Create data service
    st.session_state.data_service = DataService()
    
    # Data containers
    st.session_state.portfolio = {
        'cash': 10000.0,
        'portfolio_value': 10000.0,
        'positions': []
    }
    st.session_state.strategies = []
    st.session_state.trades = []
    st.session_state.alerts = []
    st.session_state.strategy_candidates = []
    
    # Update flag for background refresh
    st.session_state.update_requested = False
    st.session_state.bg_thread_running = False

def render_overview_page():
    """Render the portfolio overview page"""
    st.title("Portfolio Overview")
    
    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Portfolio Value",
            f"${st.session_state.portfolio['portfolio_value']:,.2f}",
            f"{get_daily_change():.2f}%"
        )
    with col2:
        st.metric(
            "Available Cash",
            f"${st.session_state.portfolio['cash']:,.2f}"
        )
    with col3:
        st.metric(
            "Daily P&L",
            f"${get_daily_pnl():,.2f}",
            f"{get_daily_pnl_percent():.2f}%"
        )
    with col4:
        st.metric(
            "Open Positions",
            f"{len(st.session_state.portfolio['positions'])}"
        )
        
    # Main content in two columns
    left_col, right_col = st.columns([3, 2])
    
    # Portfolio summary on left
    with left_col:
        render_portfolio_summary()
    
    # Recent activity on right
    with right_col:
        with st.expander("Recent Trades", expanded=True):
            # Show only the 5 most recent trades
            render_trade_log(limit=5)
        
        with st.expander("Recent Alerts", expanded=True):
            # Show only the 5 most recent alerts
            render_alert_feed(limit=5)

def render_strategies_page():
    """Render the active strategies page"""
    st.title("Active Trading Strategies")
    render_active_strategies()

def render_trade_log_page():
    """Render the full trade log page"""
    st.title("Trade Log")
    render_trade_log()

def render_alerts_page():
    """Render the full alerts page"""
    st.title("System Alerts & Events")
    render_alert_feed()

def render_strategy_lab_page():
    """Render the strategy lab page"""
    st.title("EvoTrader Strategy Lab")
    render_strategy_lab()

def trigger_background_update():
    """Trigger a background update if needed"""
    current_time = time.time()
    
    # Update data every 5 seconds if needed
    if (st.session_state.last_update_time is None or 
        current_time - st.session_state.last_update_time > 5):
        
        st.session_state.update_requested = True
        st.session_state.last_update_time = current_time
        
        # Check if we should start background thread
        if not st.session_state.bg_thread_running:
            start_background_update_thread()

def start_background_update_thread():
    """Start a background thread to update data"""
    # This function would start a thread that fetches data
    # from APIs or databases and updates session state
    # In production, this would integrate with actual BensBot components
    # For now, we simulate random data updates
    logger.info("Starting background update thread")
    st.session_state.bg_thread_running = True
    
    # Simulate data fetching in production
    st.session_state.data_service.refresh_data()

def get_daily_change():
    """Calculate daily portfolio change percentage"""
    # In a real implementation, this would fetch actual data
    daily_change = st.session_state.data_service.get_daily_change()
    return daily_change

def get_daily_pnl():
    """Get daily P&L in dollars"""
    # In a real implementation, this would fetch actual data
    daily_pnl = st.session_state.data_service.get_daily_pnl()
    return daily_pnl

def get_daily_pnl_percent():
    """Get daily P&L as percentage"""
    # In a real implementation, this would fetch actual data
    daily_pnl_pct = st.session_state.data_service.get_daily_pnl_percent()
    return daily_pnl_pct

if __name__ == "__main__":
    main()
