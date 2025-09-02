#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Monitoring Dashboard for BensBot

This dashboard provides institutional-grade oversight with four tabs:
1. Dashboard (Home) - At-a-glance overview of system status and performance
2. Performance Metrics - Detailed metrics, equity curves, and strategy allocation
3. Trade Log - Complete historical record of all trading activity
4. Approval/Audit - Controls for approving paper trading to live mode

This follows the "command bridge" metaphor - oversight, not interference.

Run with: streamlit run enhanced_monitoring_dashboard.py
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pymongo import MongoClient
import requests

# Add the project root to the Python path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Try to import the enhanced components
try:
    from trading_bot.data.persistence import PersistenceManager
    from trading_bot.core.watchdog import ServiceWatchdog, ServiceStatus
    from trading_bot.risk.capital_manager import CapitalManager
    from trading_bot.core.strategy_manager import StrategyPerformanceManager, StrategyStatus
    from trading_bot.execution.execution_model import ExecutionQualityModel
    
    COMPONENTS_IMPORTED = True
except ImportError as e:
    st.warning(f"Failed to import enhanced components: {e}")
    COMPONENTS_IMPORTED = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="BensBot Command Bridge",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize trading mode state
if "trading_mode" not in st.session_state:
    st.session_state["trading_mode"] = "PAPER MODE"  # Options: PAPER MODE, PAUSED, ACTIVE
    
if "approval_status" not in st.session_state:
    st.session_state["approval_status"] = "In Paper Trading"  # Options: In Paper Trading, Awaiting Approval, Live Trading

if "system_health_status" not in st.session_state:
    st.session_state["system_health_status"] = "GREEN"  # Options: GREEN, YELLOW, RED

# Initialize session state for connection details
if "mongodb_uri" not in st.session_state:
    st.session_state["mongodb_uri"] = "mongodb://localhost:27017/"
    
if "mongodb_database" not in st.session_state:
    st.session_state["mongodb_database"] = "bensbot"
    
if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = "http://localhost:8000"
    
if "persistence" not in st.session_state:
    st.session_state["persistence"] = None
    
if "connected" not in st.session_state:
    st.session_state["connected"] = False
    
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 10  # seconds

def connect_to_mongodb():
    """Connect to the MongoDB server using PersistenceManager"""
    try:
        uri = st.session_state["mongodb_uri"]
        database = st.session_state["mongodb_database"]
        
        # Use PersistenceManager if available, otherwise use direct connection
        if COMPONENTS_IMPORTED:
            persistence = PersistenceManager(
                connection_string=uri,
                database=database,
                auto_connect=True
            )
            st.session_state["persistence"] = persistence
            st.session_state["connected"] = persistence.is_connected()
            return persistence.is_connected()
        else:
            # Direct MongoDB connection
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')  # Check connection
            st.session_state["client"] = client
            st.session_state["db"] = client[database]
            st.session_state["connected"] = True
            return True
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        st.session_state["connected"] = False
        return False

def header():
    """Render the dashboard header"""
    st.title("BensBot Enhanced Monitoring Dashboard")
    
    # Current time and auto-refresh
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    with col2:
        st.session_state["refresh_interval"] = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=5,
            max_value=60,
            value=st.session_state["refresh_interval"],
            step=5
        )
        
    with col3:
        if st.button("Refresh Now"):
            st.experimental_rerun()

def sidebar():
    """Render the dashboard sidebar"""
    st.sidebar.title("BensBot Monitor")
    
    st.sidebar.header("Connection Settings")
    
    # MongoDB connection settings
    st.sidebar.subheader("MongoDB")
    st.session_state["mongodb_uri"] = st.sidebar.text_input(
        "MongoDB URI",
        value=st.session_state["mongodb_uri"]
    )
    
    st.session_state["mongodb_database"] = st.sidebar.text_input(
        "Database Name",
        value=st.session_state["mongodb_database"]
    )
    
    # API connection settings
    st.sidebar.subheader("API")
    st.session_state["api_base_url"] = st.sidebar.text_input(
        "API Base URL",
        value=st.session_state["api_base_url"]
    )
    
    # Connect button
    if st.sidebar.button("Connect"):
        with st.spinner("Connecting to MongoDB..."):
            if connect_to_mongodb():
                st.sidebar.success("Connected to MongoDB!")
            else:
                st.sidebar.error("Failed to connect to MongoDB")
    
    # Navigation
    st.sidebar.header("Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Persistence Monitor",
            "Watchdog Monitor",
            "Capital Management",
            "Strategy Performance",
            "Execution Quality"
        ]
    )
    
    return page

def fetch_system_health():
    """Fetch system health data from API or MongoDB"""
    try:
        # Try API first
        api_url = f"{st.session_state['api_base_url']}/health"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        # API failed, try MongoDB directly
        pass
        
    # Fallback to MongoDB for health data
    if st.session_state["connected"]:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Get logs from persistence manager
            logs_df = st.session_state["persistence"].get_system_logs(
                level="ERROR",
                limit=10
            )
            
            # Check collections stats
            health_data = {
                "status": "healthy" if st.session_state["connected"] else "unhealthy",
                "components": {
                    "persistence": "online" if st.session_state["connected"] else "offline",
                    "watchdog": "unknown",
                    "trading_engine": "unknown"
                },
                "errors": len(logs_df) if not logs_df.empty else 0,
                "last_update": datetime.now().isoformat()
            }
            
            return health_data
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            
            # Check collections
            collections = {
                "trades": db.trades.count_documents({}),
                "strategy_states": db.strategy_states.count_documents({}),
                "performance": db.performance.count_documents({}),
                "logs": db.system_logs.count_documents({})
            }
            
            # Check for recent errors
            recent_errors = db.system_logs.count_documents({
                "level": "ERROR",
                "timestamp": {"$gte": datetime.now() - timedelta(hours=1)}
            })
            
            health_data = {
                "status": "healthy" if st.session_state["connected"] else "unhealthy",
                "components": {
                    "persistence": "online" if st.session_state["connected"] else "offline",
                    "collections": collections
                },
                "errors": recent_errors,
                "last_update": datetime.now().isoformat()
            }
            
            return health_data
    
    # No connection, return dummy data
    return {
        "status": "unknown",
        "components": {},
        "errors": 0,
        "last_update": datetime.now().isoformat()
    }

def overview_page():
    """Render the overview page"""
    st.header("System Overview")
    
    # Fetch health data
    health_data = fetch_system_health()
    
    # Display health status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("System Status")
        status = health_data.get("status", "unknown")
        
        if status == "healthy":
            st.success("Healthy")
        elif status == "degraded":
            st.warning("Degraded")
        elif status == "unhealthy":
            st.error("Unhealthy")
        else:
            st.info("Unknown")
            
        st.text(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    with col2:
        st.subheader("Component Status")
        components = health_data.get("components", {})
        
        for component, status in components.items():
            if status == "online":
                st.success(f"{component.title()}: Online")
            elif status == "offline":
                st.error(f"{component.title()}: Offline")
            elif status == "degraded":
                st.warning(f"{component.title()}: Degraded")
            else:
                st.info(f"{component.title()}: Unknown")
    
    with col3:
        st.subheader("Recent Issues")
        error_count = health_data.get("errors", 0)
        
        if error_count == 0:
            st.success("No recent errors")
        else:
            st.error(f"{error_count} errors in the last hour")
    
    # Display key metrics
    st.header("Key Metrics")
    
    if st.session_state["connected"]:
        # Fetch data from MongoDB
        try:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades_df = persistence.get_trades_history(limit=1000)
                performance_df = persistence.get_performance_history(limit=100)
                
                if not trades_df.empty:
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            "Total Trades",
                            len(trades_df)
                        )
                    
                    with metric_col2:
                        # Calculate win rate
                        if 'win' in trades_df.columns:
                            win_rate = trades_df['win'].mean() * 100
                            st.metric(
                                "Win Rate",
                                f"{win_rate:.1f}%"
                            )
                        else:
                            st.metric("Win Rate", "N/A")
                    
                    with metric_col3:
                        # Calculate profit
                        if 'profit_loss' in trades_df.columns:
                            total_profit = trades_df['profit_loss'].sum()
                            st.metric(
                                "Total P&L",
                                f"${total_profit:.2f}"
                            )
                        else:
                            st.metric("Total P&L", "N/A")
                    
                    with metric_col4:
                        # Calculate active strategies
                        if 'strategy_id' in trades_df.columns:
                            active_strategies = trades_df['strategy_id'].nunique()
                            st.metric(
                                "Active Strategies",
                                active_strategies
                            )
                        else:
                            st.metric("Active Strategies", "N/A")
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                trades_count = db.trades.count_documents({})
                
                # Calculate metrics
                win_count = db.trades.count_documents({"win": True})
                total_pl = sum(trade.get("profit_loss", 0) for trade in db.trades.find({}, {"profit_loss": 1}))
                active_strategies = len(db.strategy_states.distinct("strategy_id"))
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Total Trades", trades_count)
                
                with metric_col2:
                    win_rate = (win_count / trades_count * 100) if trades_count > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with metric_col3:
                    st.metric("Total P&L", f"${total_pl:.2f}")
                
                with metric_col4:
                    st.metric("Active Strategies", active_strategies)
        except Exception as e:
            st.error(f"Error fetching metrics: {str(e)}")
    else:
        st.warning("Connect to MongoDB to view metrics")
        
        # Example charts with dummy data
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Trades", "N/A")
        
        with metric_col2:
            st.metric("Win Rate", "N/A")
        
        with metric_col3:
            st.metric("Total P&L", "N/A")
        
        with metric_col4:
            st.metric("Active Strategies", "N/A")

    # Auto-refresh functionality
    time.sleep(1)  # Small delay to prevent excessive refreshing
    st.experimental_rerun()

def persistence_page():
    """Render the persistence monitor page"""
    st.header("Persistence Layer Monitor")
    
    if not st.session_state["connected"]:
        st.warning("Not connected to MongoDB. Please connect using the sidebar.")
        return
    
    # Display connection info
    st.subheader("Connection Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"**URI:** {st.session_state['mongodb_uri']}")
        st.markdown(f"**Database:** {st.session_state['mongodb_database']}")
    
    with info_col2:
        status = "Connected" if st.session_state["connected"] else "Disconnected"
        status_color = "green" if st.session_state["connected"] else "red"
        st.markdown(f"**Status:** <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    
    # Collection statistics
    st.subheader("Collection Statistics")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            
            # Display collection stats
            col_stats = [
                {"Collection": "trades", "Documents": len(persistence.get_trades_history(limit=1000000))},
                {"Collection": "strategy_states", "Documents": len(persistence.load_strategy_state("") or [])},
                {"Collection": "performance", "Documents": len(persistence.get_performance_history(limit=1000000))},
                {"Collection": "logs", "Documents": len(persistence.get_system_logs(limit=1000000))}
            ]
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            
            # Get collection stats
            col_stats = []
            for collection_name in ["trades", "strategy_states", "performance", "system_logs"]:
                count = db[collection_name].count_documents({})
                col_stats.append({"Collection": collection_name, "Documents": count})
        
        # Display as table
        col_stats_df = pd.DataFrame(col_stats)
        st.table(col_stats_df)
    except Exception as e:
        st.error(f"Error getting collection statistics: {str(e)}")
    
    # Recent trades
    st.subheader("Recent Trades")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            trades_df = persistence.get_trades_history(limit=10)
            
            if not trades_df.empty:
                st.dataframe(trades_df)
            else:
                st.info("No trades found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            trades = list(db.trades.find().sort("timestamp", -1).limit(10))
            
            if trades:
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df)
            else:
                st.info("No trades found")
    except Exception as e:
        st.error(f"Error getting recent trades: {str(e)}")
    
    # System logs
    st.subheader("Recent System Logs")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            logs_df = persistence.get_system_logs(limit=10)
            
            if not logs_df.empty:
                st.dataframe(logs_df)
            else:
                st.info("No logs found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            logs = list(db.system_logs.find().sort("timestamp", -1).limit(10))
            
            if logs:
                logs_df = pd.DataFrame(logs)
                st.dataframe(logs_df)
            else:
                st.info("No logs found")
    except Exception as e:
        st.error(f"Error getting system logs: {str(e)}")

def watchdog_page():
    """Render the watchdog monitor page"""
    st.header("Watchdog & Fault Tolerance Monitor")
    
    # Fetch service status
    try:
        api_url = f"{st.session_state['api_base_url']}/watchdog/status"
        
        try:
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                watchdog_data = response.json()
            else:
                watchdog_data = None
        except Exception:
            watchdog_data = None
            
        # If API failed, try MongoDB
        if watchdog_data is None and st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                logs_df = st.session_state["persistence"].get_system_logs(
                    component="ServiceWatchdog",
                    limit=100
                )
                
                # Extract service status from logs
                watchdog_data = {
                    "running": True,
                    "services": {},
                    "system_health": {
                        "overall_status": "UNKNOWN",
                        "service_count": 0,
                        "uptime_seconds": 0
                    }
                }
            else:
                db = st.session_state["db"]
                logs = list(db.system_logs.find(
                    {"component": "ServiceWatchdog"}
                ).sort("timestamp", -1).limit(100))
                
                # Extract service status from logs
                watchdog_data = {
                    "running": len(logs) > 0,
                    "services": {},
                    "system_health": {
                        "overall_status": "UNKNOWN",
                        "service_count": 0,
                        "uptime_seconds": 0
                    }
                }
                
                # Extract service statuses from logs if possible
                for log in logs:
                    if "data" in log and "service" in log["data"]:
                        service_name = log["data"]["service"]
                        if service_name not in watchdog_data["services"]:
                            watchdog_data["services"][service_name] = {
                                "status": log["data"].get("status", "UNKNOWN"),
                                "last_check": log["timestamp"]
                            }
        
        # Display watchdog status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Watchdog Status")
            
            if watchdog_data:
                running = watchdog_data.get("running", False)
                status_color = "green" if running else "red"
                st.markdown(f"**Status:** <span style='color:{status_color}'>{'Running' if running else 'Stopped'}</span>", unsafe_allow_html=True)
                
                # System health
                system_health = watchdog_data.get("system_health", {})
                overall_status = system_health.get("overall_status", "UNKNOWN")
                
                if overall_status == "HEALTHY":
                    st.success(f"Overall Health: {overall_status}")
                elif overall_status == "DEGRADED":
                    st.warning(f"Overall Health: {overall_status}")
                elif overall_status == "UNHEALTHY":
                    st.error(f"Overall Health: {overall_status}")
                else:
                    st.info(f"Overall Health: {overall_status}")
                
                # Uptime
                uptime_seconds = system_health.get("uptime_seconds", 0)
                uptime_str = str(timedelta(seconds=uptime_seconds))
                st.markdown(f"**Uptime:** {uptime_str}")
                
                # Service count
                service_count = system_health.get("service_count", 0)
                st.markdown(f"**Monitored Services:** {service_count}")
            else:
                st.error("Watchdog status not available")
        
        with col2:
            st.subheader("Services Health")
            
            if watchdog_data and "services" in watchdog_data:
                services = watchdog_data["services"]
                
                for service_name, service_data in services.items():
                    status = service_data.get("status", "UNKNOWN")
                    
                    if status == "HEALTHY":
                        st.success(f"{service_name}: {status}")
                    elif status == "DEGRADED":
                        st.warning(f"{service_name}: {status}")
                    elif status == "UNHEALTHY" or status == "FAILED":
                        st.error(f"{service_name}: {status}")
                    else:
                        st.info(f"{service_name}: {status}")
            else:
                st.info("No service health data available")
        
        # Service details
        st.subheader("Service Details")
        
        if watchdog_data and "services" in watchdog_data:
            services = watchdog_data["services"]
            
            # Create table data
            table_data = []
            for service_name, service_data in services.items():
                failures = service_data.get("failures", 0)
                recovery_attempts = service_data.get("recovery_attempts", 0)
                last_failure = service_data.get("last_failure", "N/A")
                last_recovery = service_data.get("last_recovery", "N/A")
                
                table_data.append({
                    "Service": service_name,
                    "Status": service_data.get("status", "UNKNOWN"),
                    "Failures": failures,
                    "Recovery Attempts": recovery_attempts,
                    "Last Failure": last_failure,
                    "Last Recovery": last_recovery
                })
            
            if table_data:
                st.table(pd.DataFrame(table_data))
            else:
                st.info("No service details available")
        else:
            st.info("No service details available")
        
        # Recovery history
        st.subheader("Recovery History")
        
        if st.session_state["connected"]:
            try:
                if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                    # Use persistence manager
                    persistence = st.session_state["persistence"]
                    recovery_logs = persistence.get_system_logs(
                        component="ServiceWatchdog",
                        limit=20
                    )
                    
                    if not recovery_logs.empty:
                        st.dataframe(recovery_logs)
                    else:
                        st.info("No recovery history found")
                else:
                    # Direct MongoDB access
                    db = st.session_state["db"]
                    recovery_logs = list(db.system_logs.find(
                        {"component": "ServiceWatchdog", "message": {"$regex": "Recovery"}}
                    ).sort("timestamp", -1).limit(20))
                    
                    if recovery_logs:
                        st.dataframe(pd.DataFrame(recovery_logs))
                    else:
                        st.info("No recovery history found")
            except Exception as e:
                st.error(f"Error getting recovery history: {str(e)}")
    except Exception as e:
        st.error(f"Error loading watchdog data: {str(e)}")

def capital_management_page():
    """Render the capital management page"""
    st.header("Dynamic Capital Management")
    
    if not st.session_state["connected"]:
        st.warning("Not connected to MongoDB. Please connect using the sidebar.")
        return
    
    # Capital overview section
    st.subheader("Capital Overview")
    
    try:
        # Fetch capital data
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            capital_state = persistence.load_strategy_state("capital_manager")
            
            if capital_state:
                # Display capital metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    initial_capital = capital_state.get("initial_capital", 0)
                    st.metric(
                        "Initial Capital",
                        f"${initial_capital:,.2f}"
                    )
                
                with col2:
                    current_capital = capital_state.get("current_capital", 0)
                    change = current_capital - initial_capital
                    st.metric(
                        "Current Capital",
                        f"${current_capital:,.2f}",
                        delta=f"${change:+,.2f}"
                    )
                
                with col3:
                    max_capital = capital_state.get("max_capital", 0)
                    st.metric(
                        "Peak Capital",
                        f"${max_capital:,.2f}"
                    )
                
                with col4:
                    drawdown = capital_state.get("current_drawdown_pct", 0) * 100
                    max_drawdown = capital_state.get("max_drawdown_pct", 0) * 100
                    st.metric(
                        "Current Drawdown",
                        f"{drawdown:.2f}%",
                        delta=f"{max_drawdown-drawdown:.2f}% from max",
                        delta_color="inverse"
                    )
                
                # Capital history chart
                st.subheader("Capital History")
                
                # Fetch capital history
                capital_history = persistence.get_performance_history(
                    metric_type="capital",
                    limit=100
                )
                
                if not capital_history.empty:
                    # Create capital chart
                    fig = px.line(
                        capital_history,
                        x="timestamp",
                        y="value",
                        title="Capital Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No capital history data available")
            else:
                st.info("No capital manager state found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            capital_state = db.strategy_states.find_one({"strategy_id": "capital_manager"})
            
            if capital_state:
                # Display capital metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    initial_capital = capital_state.get("data", {}).get("initial_capital", 0)
                    st.metric(
                        "Initial Capital",
                        f"${initial_capital:,.2f}"
                    )
                
                with col2:
                    current_capital = capital_state.get("data", {}).get("current_capital", 0)
                    change = current_capital - initial_capital
                    st.metric(
                        "Current Capital",
                        f"${current_capital:,.2f}",
                        delta=f"${change:+,.2f}"
                    )
                
                with col3:
                    max_capital = capital_state.get("data", {}).get("max_capital", 0)
                    st.metric(
                        "Peak Capital",
                        f"${max_capital:,.2f}"
                    )
                
                with col4:
                    drawdown = capital_state.get("data", {}).get("current_drawdown_pct", 0) * 100
                    max_drawdown = capital_state.get("data", {}).get("max_drawdown_pct", 0) * 100
                    st.metric(
                        "Current Drawdown",
                        f"{drawdown:.2f}%",
                        delta=f"{max_drawdown-drawdown:.2f}% from max",
                        delta_color="inverse"
                    )
                
                # Capital history chart
                st.subheader("Capital History")
                
                # Fetch capital history
                capital_history = list(db.performance.find(
                    {"metric_type": "capital"}
                ).sort("timestamp", -1).limit(100))
                
                if capital_history:
                    # Convert to DataFrame
                    capital_df = pd.DataFrame(capital_history)
                    
                    # Create capital chart
                    fig = px.line(
                        capital_df,
                        x="timestamp",
                        y="value",
                        title="Capital Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No capital history data available")
            else:
                st.info("No capital manager state found")
    except Exception as e:
        st.error(f"Error loading capital management data: {str(e)}")
    
    # Risk Parameters
    st.subheader("Risk Management Parameters")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            capital_state = persistence.load_strategy_state("capital_manager")
            
            if capital_state and "risk_params" in capital_state:
                risk_params = capital_state["risk_params"]
                
                # Display risk parameters
                params_df = pd.DataFrame({
                    "Parameter": list(risk_params.keys()),
                    "Value": list(risk_params.values())
                })
                st.table(params_df)
            else:
                st.info("No risk parameters found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            capital_state = db.strategy_states.find_one({"strategy_id": "capital_manager"})
            
            if capital_state and "data" in capital_state and "risk_params" in capital_state["data"]:
                risk_params = capital_state["data"]["risk_params"]
                
                # Display risk parameters
                params_df = pd.DataFrame({
                    "Parameter": list(risk_params.keys()),
                    "Value": list(risk_params.values())
                })
                st.table(params_df)
            else:
                st.info("No risk parameters found")
    except Exception as e:
        st.error(f"Error loading risk parameters: {str(e)}")
    
    # Strategy Allocation
    st.subheader("Strategy Capital Allocation")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            strategy_allocations = persistence.get_performance_history(
                metric_type="strategy_allocation",
                limit=100
            )
            
            if not strategy_allocations.empty:
                # Get latest allocations
                latest_date = strategy_allocations["timestamp"].max()
                latest_allocations = strategy_allocations[strategy_allocations["timestamp"] == latest_date]
                
                # Create allocation chart
                fig = px.pie(
                    latest_allocations,
                    values="value",
                    names="strategy_id",
                    title="Current Strategy Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display allocation table
                st.dataframe(latest_allocations[["strategy_id", "value"]])
            else:
                st.info("No strategy allocation data available")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            allocations = list(db.performance.find(
                {"metric_type": "strategy_allocation"}
            ).sort("timestamp", -1))
            
            if allocations:
                # Group by latest timestamp
                allocation_df = pd.DataFrame(allocations)
                latest_date = allocation_df["timestamp"].max()
                latest_allocations = allocation_df[allocation_df["timestamp"] == latest_date]
                
                # Create allocation chart
                fig = px.pie(
                    latest_allocations,
                    values="value",
                    names="strategy_id",
                    title="Current Strategy Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display allocation table
                st.dataframe(latest_allocations[["strategy_id", "value"]])
            else:
                st.info("No strategy allocation data available")
    except Exception as e:
        st.error(f"Error loading strategy allocations: {str(e)}")
    
    # Capital scaling factors
    st.subheader("Dynamic Scaling Factors")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            scaling_factors = persistence.get_performance_history(
                metric_type="scaling_factor",
                limit=100
            )
            
            if not scaling_factors.empty:
                # Create scaling factors chart
                fig = px.line(
                    scaling_factors,
                    x="timestamp",
                    y="value",
                    color="factor_type",
                    title="Dynamic Scaling Factors Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scaling factor data available")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            factors = list(db.performance.find(
                {"metric_type": "scaling_factor"}
            ).sort("timestamp", -1).limit(100))
            
            if factors:
                # Convert to DataFrame
                factors_df = pd.DataFrame(factors)
                
                # Create scaling factors chart
                fig = px.line(
                    factors_df,
                    x="timestamp",
                    y="value",
                    color="factor_type",
                    title="Dynamic Scaling Factors Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scaling factor data available")
    except Exception as e:
        st.error(f"Error loading scaling factors: {str(e)}")

def strategy_performance_page():
    """Render the strategy performance page"""
    st.header("Strategy Performance & Lifecycle")
    
    if not st.session_state["connected"]:
        st.warning("Not connected to MongoDB. Please connect using the sidebar.")
        return
    
    # Strategy overview section
    st.subheader("Strategy Overview")
    
    try:
        # Fetch active strategies
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            strategy_state = persistence.load_strategy_state("strategy_manager")
            
            if strategy_state and "strategies" in strategy_state:
                strategies = strategy_state["strategies"]
                
                # Count strategies by status
                status_counts = {}
                for strategy_id, strategy_data in strategies.items():
                    status = strategy_data.get("status", "UNKNOWN")
                    if status not in status_counts:
                        status_counts[status] = 0
                    status_counts[status] += 1
                
                # Display strategy status counts
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    active_count = status_counts.get("ACTIVE", 0)
                    st.metric(
                        "Active Strategies",
                        active_count
                    )
                
                with col2:
                    testing_count = status_counts.get("TESTING", 0)
                    st.metric(
                        "Testing Strategies",
                        testing_count
                    )
                
                with col3:
                    retired_count = status_counts.get("RETIRED", 0)
                    st.metric(
                        "Retired Strategies",
                        retired_count
                    )
                
                with col4:
                    promoted_count = status_counts.get("PROMOTED", 0)
                    st.metric(
                        "Promoted Strategies",
                        promoted_count
                    )
                
                # Display strategy status pie chart
                status_df = pd.DataFrame({
                    "Status": list(status_counts.keys()),
                    "Count": list(status_counts.values())
                })
                
                fig = px.pie(
                    status_df,
                    values="Count",
                    names="Status",
                    title="Strategy Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy manager state found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            strategy_state = db.strategy_states.find_one({"strategy_id": "strategy_manager"})
            
            if strategy_state and "data" in strategy_state and "strategies" in strategy_state["data"]:
                strategies = strategy_state["data"]["strategies"]
                
                # Count strategies by status
                status_counts = {}
                for strategy_id, strategy_data in strategies.items():
                    status = strategy_data.get("status", "UNKNOWN")
                    if status not in status_counts:
                        status_counts[status] = 0
                    status_counts[status] += 1
                
                # Display strategy status counts
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    active_count = status_counts.get("ACTIVE", 0)
                    st.metric(
                        "Active Strategies",
                        active_count
                    )
                
                with col2:
                    testing_count = status_counts.get("TESTING", 0)
                    st.metric(
                        "Testing Strategies",
                        testing_count
                    )
                
                with col3:
                    retired_count = status_counts.get("RETIRED", 0)
                    st.metric(
                        "Retired Strategies",
                        retired_count
                    )
                
                with col4:
                    promoted_count = status_counts.get("PROMOTED", 0)
                    st.metric(
                        "Promoted Strategies",
                        promoted_count
                    )
                
                # Display strategy status pie chart
                status_df = pd.DataFrame({
                    "Status": list(status_counts.keys()),
                    "Count": list(status_counts.values())
                })
                
                fig = px.pie(
                    status_df,
                    values="Count",
                    names="Status",
                    title="Strategy Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy manager state found")
    except Exception as e:
        st.error(f"Error loading strategy overview: {str(e)}")
    
    # Strategy performance metrics
    st.subheader("Strategy Performance Metrics")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            strategy_metrics = persistence.get_performance_history(
                metric_type="strategy_metrics",
                limit=1000
            )
            
            if not strategy_metrics.empty:
                # Group by strategy
                strategies = strategy_metrics["strategy_id"].unique()
                
                # Create tabs for each strategy
                tabs = st.tabs([f"Strategy: {strategy}" for strategy in strategies])
                
                for i, strategy in enumerate(strategies):
                    with tabs[i]:
                        # Filter metrics for this strategy
                        strategy_data = strategy_metrics[strategy_metrics["strategy_id"] == strategy]
                        
                        # Create metrics sections
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        # Get latest metrics
                        latest_date = strategy_data["timestamp"].max()
                        latest_metrics = strategy_data[strategy_data["timestamp"] == latest_date]
                        
                        with metric_col1:
                            win_rate = latest_metrics[latest_metrics["metric_name"] == "win_rate"]["value"].values
                            if len(win_rate) > 0:
                                st.metric(
                                    "Win Rate",
                                    f"{win_rate[0]:.2f}%"
                                )
                            else:
                                st.metric("Win Rate", "N/A")
                        
                        with metric_col2:
                            sharpe = latest_metrics[latest_metrics["metric_name"] == "sharpe_ratio"]["value"].values
                            if len(sharpe) > 0:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{sharpe[0]:.2f}"
                                )
                            else:
                                st.metric("Sharpe Ratio", "N/A")
                        
                        with metric_col3:
                            profit_factor = latest_metrics[latest_metrics["metric_name"] == "profit_factor"]["value"].values
                            if len(profit_factor) > 0:
                                st.metric(
                                    "Profit Factor",
                                    f"{profit_factor[0]:.2f}"
                                )
                            else:
                                st.metric("Profit Factor", "N/A")
                        
                        with metric_col4:
                            drawdown = latest_metrics[latest_metrics["metric_name"] == "max_drawdown"]["value"].values
                            if len(drawdown) > 0:
                                st.metric(
                                    "Max Drawdown",
                                    f"{drawdown[0]:.2f}%"
                                )
                            else:
                                st.metric("Max Drawdown", "N/A")
                        
                        # Plot metrics over time
                        st.subheader(f"Metrics Over Time - {strategy}")
                        
                        # Pivot the data to get metrics by timestamp
                        pivoted = strategy_data.pivot(index="timestamp", columns="metric_name", values="value")
                        pivoted.reset_index(inplace=True)
                        
                        # Create line chart
                        fig = px.line(
                            pivoted,
                            x="timestamp",
                            y=pivoted.columns[1:],  # Skip the timestamp column
                            title=f"{strategy} - Performance Metrics Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy metrics data available")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            metrics = list(db.performance.find(
                {"metric_type": "strategy_metrics"}
            ).sort("timestamp", -1).limit(1000))
            
            if metrics:
                # Convert to DataFrame
                metrics_df = pd.DataFrame(metrics)
                
                # Group by strategy
                strategies = metrics_df["strategy_id"].unique()
                
                # Create tabs for each strategy
                tabs = st.tabs([f"Strategy: {strategy}" for strategy in strategies])
                
                for i, strategy in enumerate(strategies):
                    with tabs[i]:
                        # Filter metrics for this strategy
                        strategy_data = metrics_df[metrics_df["strategy_id"] == strategy]
                        
                        # Create metrics sections
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        # Get latest metrics
                        latest_date = strategy_data["timestamp"].max()
                        latest_metrics = strategy_data[strategy_data["timestamp"] == latest_date]
                        
                        with metric_col1:
                            win_rate = latest_metrics[latest_metrics["metric_name"] == "win_rate"]["value"].values
                            if len(win_rate) > 0:
                                st.metric(
                                    "Win Rate",
                                    f"{win_rate[0]:.2f}%"
                                )
                            else:
                                st.metric("Win Rate", "N/A")
                        
                        with metric_col2:
                            sharpe = latest_metrics[latest_metrics["metric_name"] == "sharpe_ratio"]["value"].values
                            if len(sharpe) > 0:
                                st.metric(
                                    "Sharpe Ratio",
                                    f"{sharpe[0]:.2f}"
                                )
                            else:
                                st.metric("Sharpe Ratio", "N/A")
                        
                        with metric_col3:
                            profit_factor = latest_metrics[latest_metrics["metric_name"] == "profit_factor"]["value"].values
                            if len(profit_factor) > 0:
                                st.metric(
                                    "Profit Factor",
                                    f"{profit_factor[0]:.2f}"
                                )
                            else:
                                st.metric("Profit Factor", "N/A")
                        
                        with metric_col4:
                            drawdown = latest_metrics[latest_metrics["metric_name"] == "max_drawdown"]["value"].values
                            if len(drawdown) > 0:
                                st.metric(
                                    "Max Drawdown",
                                    f"{drawdown[0]:.2f}%"
                                )
                            else:
                                st.metric("Max Drawdown", "N/A")
                        
                        # Plot metrics over time
                        st.subheader(f"Metrics Over Time - {strategy}")
                        
                        # Pivot the data to get metrics by timestamp
                        pivoted = strategy_data.pivot(index="timestamp", columns="metric_name", values="value")
                        pivoted.reset_index(inplace=True)
                        
                        # Create line chart
                        fig = px.line(
                            pivoted,
                            x="timestamp",
                            y=pivoted.columns[1:],  # Skip the timestamp column
                            title=f"{strategy} - Performance Metrics Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy metrics data available")
    except Exception as e:
        st.error(f"Error loading strategy metrics: {str(e)}")
    
    # Strategy Lifecycle Events
    st.subheader("Strategy Lifecycle Events")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            lifecycle_events = persistence.get_system_logs(
                component="StrategyPerformanceManager",
                level="INFO",
                limit=100
            )
            
            if not lifecycle_events.empty:
                # Filter for lifecycle events
                lifecycle_events = lifecycle_events[
                    lifecycle_events["message"].str.contains("RETIRED|PROMOTED|TESTING|ACTIVE")
                ]
                
                # Display events
                st.dataframe(lifecycle_events[["timestamp", "message", "strategy_id", "data"]])
            else:
                st.info("No strategy lifecycle events found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            lifecycle_events = list(db.system_logs.find(
                {
                    "component": "StrategyPerformanceManager",
                    "message": {"$regex": "RETIRED|PROMOTED|TESTING|ACTIVE"}
                }
            ).sort("timestamp", -1).limit(100))
            
            if lifecycle_events:
                # Convert to DataFrame
                events_df = pd.DataFrame(lifecycle_events)
                
                # Display events
                st.dataframe(events_df[["timestamp", "message", "strategy_id", "data"]])
            else:
                st.info("No strategy lifecycle events found")
    except Exception as e:
        st.error(f"Error loading lifecycle events: {str(e)}")
    
    # Strategy Performance Rankings
    st.subheader("Strategy Performance Rankings")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            strategy_state = persistence.load_strategy_state("strategy_manager")
            
            if strategy_state and "performance_rankings" in strategy_state:
                rankings = strategy_state["performance_rankings"]
                
                # Create a DataFrame from rankings
                rankings_data = []
                for metric, ranked_strategies in rankings.items():
                    for rank, (strategy_id, value) in enumerate(ranked_strategies):
                        rankings_data.append({
                            "Metric": metric,
                            "Rank": rank + 1,
                            "Strategy": strategy_id,
                            "Value": value
                        })
                
                rankings_df = pd.DataFrame(rankings_data)
                
                # Create a tab for each metric
                metrics = rankings_df["Metric"].unique()
                tabs = st.tabs([f"Ranking: {metric}" for metric in metrics])
                
                for i, metric in enumerate(metrics):
                    with tabs[i]:
                        # Filter for this metric
                        metric_data = rankings_df[rankings_df["Metric"] == metric]
                        
                        # Sort by rank
                        metric_data = metric_data.sort_values("Rank")
                        
                        # Display as bar chart
                        fig = px.bar(
                            metric_data,
                            x="Strategy",
                            y="Value",
                            title=f"Strategy Ranking by {metric}",
                            color="Rank",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display as table
                        st.dataframe(metric_data[["Rank", "Strategy", "Value"]].set_index("Rank"))
            else:
                st.info("No strategy performance rankings found")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            strategy_state = db.strategy_states.find_one({"strategy_id": "strategy_manager"})
            
            if strategy_state and "data" in strategy_state and "performance_rankings" in strategy_state["data"]:
                rankings = strategy_state["data"]["performance_rankings"]
                
                # Create a DataFrame from rankings
                rankings_data = []
                for metric, ranked_strategies in rankings.items():
                    for rank, (strategy_id, value) in enumerate(ranked_strategies):
                        rankings_data.append({
                            "Metric": metric,
                            "Rank": rank + 1,
                            "Strategy": strategy_id,
                            "Value": value
                        })
                
                rankings_df = pd.DataFrame(rankings_data)
                
                # Create a tab for each metric
                metrics = rankings_df["Metric"].unique()
                tabs = st.tabs([f"Ranking: {metric}" for metric in metrics])
                
                for i, metric in enumerate(metrics):
                    with tabs[i]:
                        # Filter for this metric
                        metric_data = rankings_df[rankings_df["Metric"] == metric]
                        
                        # Sort by rank
                        metric_data = metric_data.sort_values("Rank")
                        
                        # Display as bar chart
                        fig = px.bar(
                            metric_data,
                            x="Strategy",
                            y="Value",
                            title=f"Strategy Ranking by {metric}",
                            color="Rank",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display as table
                        st.dataframe(metric_data[["Rank", "Strategy", "Value"]].set_index("Rank"))
            else:
                st.info("No strategy performance rankings found")
    except Exception as e:
        st.error(f"Error loading strategy rankings: {str(e)}")

def execution_quality_page():
    """Render the execution quality page"""
    st.header("Execution Quality Metrics")
    
    if not st.session_state["connected"]:
        st.warning("Not connected to MongoDB. Please connect using the sidebar.")
        return
    
    # Execution overview section
    st.subheader("Execution Overview")
    
    try:
        # Fetch execution data
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            execution_metrics = persistence.get_performance_history(
                metric_type="execution_quality",
                limit=1000
            )
            
            if not execution_metrics.empty:
                # Display execution metrics summary
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate mean metrics
                mean_slippage = execution_metrics[execution_metrics["metric_name"] == "slippage"]["value"].mean()
                mean_latency = execution_metrics[execution_metrics["metric_name"] == "latency"]["value"].mean()
                mean_spread = execution_metrics[execution_metrics["metric_name"] == "effective_spread"]["value"].mean()
                mean_impact = execution_metrics[execution_metrics["metric_name"] == "market_impact"]["value"].mean()
                
                with col1:
                    st.metric(
                        "Avg. Slippage (pips)",
                        f"{mean_slippage:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Avg. Latency (ms)",
                        f"{mean_latency:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Avg. Spread (pips)",
                        f"{mean_spread:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Avg. Market Impact (pips)",
                        f"{mean_impact:.2f}"
                    )
                
                # Execution metrics by symbol
                st.subheader("Execution Metrics by Symbol")
                
                # Group by symbol
                symbols = execution_metrics["symbol"].unique()
                
                # Create tabs for different metric types
                metric_tabs = st.tabs(["Slippage", "Latency", "Spread", "Market Impact"])
                
                with metric_tabs[0]:  # Slippage
                    slippage_data = execution_metrics[execution_metrics["metric_name"] == "slippage"]
                    
                    if not slippage_data.empty:
                        # Group by symbol
                        symbol_slippage = slippage_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_slippage,
                            x="symbol",
                            y="value",
                            title="Average Slippage by Symbol (pips)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high slippage, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No slippage data available")
                
                with metric_tabs[1]:  # Latency
                    latency_data = execution_metrics[execution_metrics["metric_name"] == "latency"]
                    
                    if not latency_data.empty:
                        # Group by symbol
                        symbol_latency = latency_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_latency,
                            x="symbol",
                            y="value",
                            title="Average Latency by Symbol (ms)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high latency, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No latency data available")
                
                with metric_tabs[2]:  # Spread
                    spread_data = execution_metrics[execution_metrics["metric_name"] == "effective_spread"]
                    
                    if not spread_data.empty:
                        # Group by symbol
                        symbol_spread = spread_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_spread,
                            x="symbol",
                            y="value",
                            title="Average Spread by Symbol (pips)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high spread, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No spread data available")
                
                with metric_tabs[3]:  # Market Impact
                    impact_data = execution_metrics[execution_metrics["metric_name"] == "market_impact"]
                    
                    if not impact_data.empty:
                        # Group by symbol
                        symbol_impact = impact_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_impact,
                            x="symbol",
                            y="value",
                            title="Average Market Impact by Symbol (pips)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high impact, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No market impact data available")
                
                # Execution trends over time
                st.subheader("Execution Trends")
                
                # Create line chart for trends over time
                fig = px.line(
                    execution_metrics,
                    x="timestamp",
                    y="value",
                    color="metric_name",
                    facet_col="metric_name",
                    facet_col_wrap=2,  # 2 charts per row
                    title="Execution Metrics Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No execution quality metrics available")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            execution_metrics = list(db.performance.find(
                {"metric_type": "execution_quality"}
            ).sort("timestamp", -1).limit(1000))
            
            if execution_metrics:
                # Convert to DataFrame
                metrics_df = pd.DataFrame(execution_metrics)
                
                # Display execution metrics summary
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate mean metrics
                mean_slippage = metrics_df[metrics_df["metric_name"] == "slippage"]["value"].mean()
                mean_latency = metrics_df[metrics_df["metric_name"] == "latency"]["value"].mean()
                mean_spread = metrics_df[metrics_df["metric_name"] == "effective_spread"]["value"].mean()
                mean_impact = metrics_df[metrics_df["metric_name"] == "market_impact"]["value"].mean()
                
                with col1:
                    st.metric(
                        "Avg. Slippage (pips)",
                        f"{mean_slippage:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Avg. Latency (ms)",
                        f"{mean_latency:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Avg. Spread (pips)",
                        f"{mean_spread:.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Avg. Market Impact (pips)",
                        f"{mean_impact:.2f}"
                    )
                
                # Execution metrics by symbol
                st.subheader("Execution Metrics by Symbol")
                
                # Group by symbol
                symbols = metrics_df["symbol"].unique()
                
                # Create tabs for different metric types
                metric_tabs = st.tabs(["Slippage", "Latency", "Spread", "Market Impact"])
                
                with metric_tabs[0]:  # Slippage
                    slippage_data = metrics_df[metrics_df["metric_name"] == "slippage"]
                    
                    if not slippage_data.empty:
                        # Group by symbol
                        symbol_slippage = slippage_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_slippage,
                            x="symbol",
                            y="value",
                            title="Average Slippage by Symbol (pips)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high slippage, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No slippage data available")
                
                with metric_tabs[1]:  # Latency
                    latency_data = metrics_df[metrics_df["metric_name"] == "latency"]
                    
                    if not latency_data.empty:
                        # Group by symbol
                        symbol_latency = latency_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_latency,
                            x="symbol",
                            y="value",
                            title="Average Latency by Symbol (ms)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high latency, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No latency data available")
                
                with metric_tabs[2]:  # Spread
                    spread_data = metrics_df[metrics_df["metric_name"] == "effective_spread"]
                    
                    if not spread_data.empty:
                        # Group by symbol
                        symbol_spread = spread_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_spread,
                            x="symbol",
                            y="value",
                            title="Average Spread by Symbol (pips)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high spread, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No spread data available")
                
                with metric_tabs[3]:  # Market Impact
                    impact_data = metrics_df[metrics_df["metric_name"] == "market_impact"]
                    
                    if not impact_data.empty:
                        # Group by symbol
                        symbol_impact = impact_data.groupby("symbol")["value"].mean().reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            symbol_impact,
                            x="symbol",
                            y="value",
                            title="Average Market Impact by Symbol (pips)",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high impact, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No market impact data available")
                
                # Execution trends over time
                st.subheader("Execution Trends")
                
                # Create line chart for trends over time
                fig = px.line(
                    metrics_df,
                    x="timestamp",
                    y="value",
                    color="metric_name",
                    facet_col="metric_name",
                    facet_col_wrap=2,  # 2 charts per row
                    title="Execution Metrics Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No execution quality metrics available")
    except Exception as e:
        st.error(f"Error loading execution quality metrics: {str(e)}")
    
    # Order fill quality
    st.subheader("Order Fill Quality")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            trades = persistence.get_trades_history(limit=1000)
            
            if not trades.empty and "fill_quality" in trades.columns:
                # Calculate fill quality metrics
                avg_fill_quality = trades["fill_quality"].mean()
                
                st.metric(
                    "Average Fill Quality",
                    f"{avg_fill_quality:.2f}%"
                )
                
                # Group by symbol
                fill_by_symbol = trades.groupby("symbol")["fill_quality"].mean().reset_index()
                
                # Create bar chart
                fig = px.bar(
                    fill_by_symbol,
                    x="symbol",
                    y="fill_quality",
                    title="Fill Quality by Symbol (%)",
                    color="fill_quality",
                    color_continuous_scale="Viridis"  # Purple for higher quality
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Fill quality vs. volume scatter plot
                if "volume" in trades.columns:
                    fig = px.scatter(
                        trades,
                        x="volume",
                        y="fill_quality",
                        color="symbol",
                        size="volume",
                        hover_data=["timestamp", "price"],
                        title="Fill Quality vs. Order Volume"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fill quality data available in trades")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            trades = list(db.trades.find().sort("timestamp", -1).limit(1000))
            
            if trades:
                trades_df = pd.DataFrame(trades)
                
                if "fill_quality" in trades_df.columns:
                    # Calculate fill quality metrics
                    avg_fill_quality = trades_df["fill_quality"].mean()
                    
                    st.metric(
                        "Average Fill Quality",
                        f"{avg_fill_quality:.2f}%"
                    )
                    
                    # Group by symbol
                    fill_by_symbol = trades_df.groupby("symbol")["fill_quality"].mean().reset_index()
                    
                    # Create bar chart
                    fig = px.bar(
                        fill_by_symbol,
                        x="symbol",
                        y="fill_quality",
                        title="Fill Quality by Symbol (%)",
                        color="fill_quality",
                        color_continuous_scale="Viridis"  # Purple for higher quality
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fill quality vs. volume scatter plot
                    if "volume" in trades_df.columns:
                        fig = px.scatter(
                            trades_df,
                            x="volume",
                            y="fill_quality",
                            color="symbol",
                            size="volume",
                            hover_data=["timestamp", "price"],
                            title="Fill Quality vs. Order Volume"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fill quality data available in trades")
            else:
                st.info("No trades data available")
    except Exception as e:
        st.error(f"Error loading order fill quality data: {str(e)}")
    
    # Session Analysis
    st.subheader("Execution Quality by Session")
    
    try:
        if COMPONENTS_IMPORTED and st.session_state["persistence"]:
            # Use persistence manager
            persistence = st.session_state["persistence"]
            execution_metrics = persistence.get_performance_history(
                metric_type="execution_quality",
                limit=1000
            )
            
            if not execution_metrics.empty and "session" in execution_metrics.columns:
                # Group by session and metric name
                session_metrics = execution_metrics.groupby(["session", "metric_name"])["value"].mean().reset_index()
                
                # Create bar chart for each metric by session
                for metric in session_metrics["metric_name"].unique():
                    metric_data = session_metrics[session_metrics["metric_name"] == metric]
                    
                    fig = px.bar(
                        metric_data,
                        x="session",
                        y="value",
                        title=f"Average {metric.capitalize()} by Trading Session",
                        color="value",
                        color_continuous_scale="RdYlGn_r"  # Red for high metrics, green for low
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No session-based execution data available")
        else:
            # Direct MongoDB access
            db = st.session_state["db"]
            execution_metrics = list(db.performance.find(
                {"metric_type": "execution_quality"}
            ).limit(1000))
            
            if execution_metrics:
                metrics_df = pd.DataFrame(execution_metrics)
                
                if "session" in metrics_df.columns:
                    # Group by session and metric name
                    session_metrics = metrics_df.groupby(["session", "metric_name"])["value"].mean().reset_index()
                    
                    # Create bar chart for each metric by session
                    for metric in session_metrics["metric_name"].unique():
                        metric_data = session_metrics[session_metrics["metric_name"] == metric]
                        
                        fig = px.bar(
                            metric_data,
                            x="session",
                            y="value",
                            title=f"Average {metric.capitalize()} by Trading Session",
                            color="value",
                            color_continuous_scale="RdYlGn_r"  # Red for high metrics, green for low
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No session data available in execution metrics")
            else:
                st.info("No execution metrics data available")
    except Exception as e:
        st.error(f"Error loading session-based execution data: {str(e)}")

def dashboard_home_tab():
    """Dashboard (Home) tab - At-a-glance view of system status and performance"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Trading Status
        st.subheader("Trading Status")
        status_color = {
            "ACTIVE": "green",
            "PAUSED": "orange",
            "PAPER MODE": "blue"
        }.get(st.session_state["trading_mode"], "gray")
        st.markdown(f"<h3 style='color:{status_color};'>{st.session_state['trading_mode']}</h3>", unsafe_allow_html=True)
        
        # System Health Indicator
        st.subheader("System Health")
        health_color = {
            "GREEN": "green",
            "YELLOW": "orange",
            "RED": "red"
        }.get(st.session_state["system_health_status"], "gray")
        st.markdown(f"<h3 style='color:{health_color};'>{st.session_state['system_health_status']}</h3>", unsafe_allow_html=True)
    
    with col2:
        # PnL Summary
        st.subheader("PnL Summary")
        
        # Get PnL data from persistence
        try:
            if st.session_state["connected"]:
                # Today's PnL
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                today_pnl = get_pnl_since(today)
                st.metric("Today's PnL", f"${today_pnl:.2f}")
                
                # Cumulative Paper Test PnL
                paper_pnl = get_total_pnl(mode="paper")
                st.metric("Cumulative Paper Test PnL", f"${paper_pnl:.2f}")
                
                # Live PnL (if in live mode)
                if st.session_state["trading_mode"] == "ACTIVE":
                    live_pnl = get_total_pnl(mode="live")
                    st.metric("Live PnL", f"${live_pnl:.2f}")
        except Exception as e:
            st.error(f"Error getting PnL data: {str(e)}")
    
    with col3:
        # Quick Stats Panel
        st.subheader("Quick Stats")
        
        try:
            if st.session_state["connected"]:
                # Win rate
                win_rate = get_win_rate()
                st.metric("Win Rate", f"{win_rate:.2f}%")
                
                # Average trade gain/loss
                avg_gain = get_average_trade_gain()
                st.metric("Avg Trade Gain/Loss", f"{avg_gain:.2f}%")
                
                # Sharpe ratio
                sharpe = get_sharpe_ratio()
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                # Drawdown
                drawdown = get_current_drawdown()
                st.metric("Drawdown", f"{drawdown:.2f}%")
        except Exception as e:
            st.error(f"Error getting stats data: {str(e)}")
    
    # Approval Status
    st.subheader("Approval Status")
    st.info(st.session_state["approval_status"])
    
    # System Health Details - From our existing watchdog_page functionality
    st.subheader("System Health Details")
    try:
        api_url = f"{st.session_state['api_base_url']}/watchdog/status"
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                watchdog_data = response.json()
                
                # Display service statuses in a table
                if "services" in watchdog_data and watchdog_data["services"]:
                    services = watchdog_data["services"]
                    service_data = []
                    
                    for service_id, details in services.items():
                        status_color = {
                            "HEALTHY": "green",
                            "DEGRADED": "orange",
                            "UNHEALTHY": "red",
                            "UNKNOWN": "gray"
                        }.get(details.get("status", "UNKNOWN"), "gray")
                        
                        service_data.append({
                            "Service": service_id,
                            "Status": details.get("status", "UNKNOWN"),
                            "Last Check": details.get("last_check_time", ""),
                            "Failures": details.get("failure_count", 0),
                            "Status Color": status_color
                        })
                    
                    # Update the overall system health based on service statuses
                    if any(s["Status"] == "UNHEALTHY" for s in service_data):
                        st.session_state["system_health_status"] = "RED"
                    elif any(s["Status"] == "DEGRADED" for s in service_data):
                        st.session_state["system_health_status"] = "YELLOW"
                    elif all(s["Status"] == "HEALTHY" for s in service_data):
                        st.session_state["system_health_status"] = "GREEN"
                    
                    # Display as dataframe with colored statuses
                    df = pd.DataFrame(service_data)
                    st.dataframe(df)
            else:
                st.warning("Could not fetch service status from API")
        except Exception:
            st.warning("API connection failed")
    except Exception as e:
        st.error(f"Error getting system health data: {str(e)}")

def performance_metrics_tab():
    """Performance Metrics tab - Detailed metrics and visualizations"""
    # Combine elements from capital_management_page and strategy_performance_page
    st.header("Performance Metrics")
    
    # Create tabs for basic metrics and strategy intelligence
    basic_tab, intelligence_tab = st.tabs(["Basic Metrics", "Strategy Intelligence"])
    
    with basic_tab:
        # Equity curve chart
        st.subheader("Equity Curve")
        try:
            if st.session_state["connected"]:
                # Get equity curve data
                equity_data = get_equity_curve_data()
                if equity_data is not None and not equity_data.empty:
                    # Create paper vs live comparison if both exist
                    paper_data = equity_data[equity_data["mode"] == "paper"]
                    live_data = equity_data[equity_data["mode"] == "live"]
                    
                    fig = go.Figure()
                    if not paper_data.empty:
                        fig.add_trace(go.Scatter(
                            x=paper_data["timestamp"],
                            y=paper_data["equity"],
                            mode="lines",
                            name="Paper Trading"
                        ))
                    
                    if not live_data.empty:
                        fig.add_trace(go.Scatter(
                            x=live_data["timestamp"],
                            y=live_data["equity"],
                            mode="lines",
                            name="Live Trading"
                        ))
                    
                    fig.update_layout(title="Equity Curve: Paper vs Live")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No equity curve data available")
        except Exception as e:
            st.error(f"Error loading equity curve data: {str(e)}")
        
        # Trade distribution
        st.subheader("Trade Distribution")
        try:
            if st.session_state["connected"]:
                # Get trade distribution data
                win_loss = get_win_loss_distribution()
                if win_loss is not None:
                    fig = px.pie(
                        values=[win_loss["wins"], win_loss["losses"]],
                        names=["Wins", "Losses"],
                        title="Win/Loss Distribution",
                        color_discrete_sequence=["green", "red"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trade distribution data available")
        except Exception as e:
            st.error(f"Error loading trade distribution data: {str(e)}")
        
        # Strategy allocation
        st.subheader("Strategy Allocation")
        try:
            if st.session_state["connected"]:
                # Get strategy allocation data
                strategy_data = get_strategy_allocation()
                if strategy_data is not None and not strategy_data.empty:
                    fig = px.pie(
                        strategy_data,
                        values="allocation",
                        names="strategy",
                        title="Capital Allocation by Strategy",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strategy allocation data available")
        except Exception as e:
            st.error(f"Error loading strategy allocation data: {str(e)}")
    
    with intelligence_tab:
        st.header("Strategy Intelligence")
        st.write("Insights into the decision-making process of the trading system")
        
        # Create expandable sections for each intelligence component
        with st.expander("Asset Selection Rationale", expanded=True):
            st.subheader("Asset Selection Rationale")
            
            # Market analysis that led to asset class selection
            st.write("### Market Analysis")
            try:
                market_analysis = get_market_analysis_data()
                if market_analysis is not None and not market_analysis.empty:
                    # Show asset class opportunities
                    fig = px.bar(
                        market_analysis,
                        x="asset_class", 
                        y="opportunity_score",
                        color="opportunity_score",
                        title="Opportunity Score by Asset Class",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show reasoning behind selections
                    st.write("### Selection Reasoning")
                    st.dataframe(market_analysis[["asset_class", "opportunity_score", "selection_reason"]],
                               use_container_width=True)
                else:
                    st.info("No asset selection data available")
            except Exception as e:
                st.error(f"Error loading asset selection data: {str(e)}")
            
            # Correlation analysis between selected assets
            st.write("### Asset Correlation")
            try:
                correlation_data = get_asset_correlation_data()
                if correlation_data is not None:
                    fig = px.imshow(
                        correlation_data,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Asset Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No correlation data available")
            except Exception as e:
                st.error(f"Error loading correlation data: {str(e)}")
        
        with st.expander("Symbol Selection Logic", expanded=False):
            st.subheader("Symbol Selection Logic")
            
            # Ranking criteria used for symbol selection
            st.write("### Symbol Ranking Criteria")
            try:
                symbol_rankings = get_symbol_ranking_data()
                if symbol_rankings is not None and not symbol_rankings.empty:
                    # Show ranking table
                    st.dataframe(symbol_rankings, use_container_width=True)
                    
                    # Visualize key metrics for top symbols
                    top_symbols = symbol_rankings.sort_values("total_score", ascending=False).head(5)
                    
                    # Radar chart of metrics
                    categories = ["liquidity", "volatility", "spread", "trend_strength", "regime_fit"]
                    fig = go.Figure()
                    
                    for i, (_, row) in enumerate(top_symbols.iterrows()):
                        fig.add_trace(go.Scatterpolar(
                            r=[row[cat] for cat in categories],
                            theta=categories,
                            fill='toself',
                            name=row["symbol"]
                        ))
                        
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        title="Top Symbols - Key Metrics"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No symbol ranking data available")
            except Exception as e:
                st.error(f"Error loading symbol ranking data: {str(e)}")
            
            # Historical performance in similar conditions
            st.write("### Historical Performance in Similar Conditions")
            try:
                historical_perf = get_symbol_historical_performance()
                if historical_perf is not None and not historical_perf.empty:
                    # Show historical performance heatmap
                    pivot_data = historical_perf.pivot("symbol", "market_regime", "performance")
                    fig = px.imshow(
                        pivot_data,
                        text_auto=True,
                        color_continuous_scale="RdYlGn",
                        title="Symbol Performance by Market Regime"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No historical performance data available")
            except Exception as e:
                st.error(f"Error loading historical performance data: {str(e)}")
        
        with st.expander("Strategy Selection Insights", expanded=False):
            st.subheader("Strategy Selection Insights")
            
            # Market regime detection results
            st.write("### Current Market Regime")
            try:
                regime_data = get_market_regime_data()
                if regime_data is not None:
                    # Market regime gauge
                    current_regime = regime_data.get("current_regime", "Unknown")
                    confidence = regime_data.get("confidence", 0) * 100
                    
                    # Create gauge chart for regime confidence
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={"text": f"Current Regime: {current_regime}"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 75], "color": "gray"},
                                {"range": [75, 100], "color": "darkblue"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show regime history
                    if "history" in regime_data:
                        regime_history_df = pd.DataFrame(regime_data["history"])
                        st.write("### Recent Regime Changes")
                        st.dataframe(regime_history_df, use_container_width=True)
                else:
                    st.info("No market regime data available")
            except Exception as e:
                st.error(f"Error loading market regime data: {str(e)}")
            
            # Strategy compatibility matrix
            st.write("### Strategy Compatibility Matrix")
            try:
                compatibility_data = get_strategy_compatibility_data()
                if compatibility_data is not None and not compatibility_data.empty:
                    # Heatmap of strategy compatibility with current regime
                    fig = px.imshow(
                        compatibility_data,
                        text_auto=True,
                        color_continuous_scale="Viridis",
                        title="Strategy Compatibility with Market Regimes",
                        labels={"x": "Market Regime", "y": "Strategy", "color": "Score"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show performance benchmarks
                    st.write("### Strategy Performance in Similar Regimes")
                    perf_benchmarks = get_strategy_performance_benchmarks()
                    if perf_benchmarks is not None and not perf_benchmarks.empty:
                        st.dataframe(perf_benchmarks, use_container_width=True)
                else:
                    st.info("No strategy compatibility data available")
            except Exception as e:
                st.error(f"Error loading strategy compatibility data: {str(e)}")
        
        with st.expander("Performance Attribution", expanded=False):
            st.subheader("Performance Attribution")
            
            # Components that contributed to performance
            st.write("### Performance Breakdown")
            try:
                attribution_data = get_performance_attribution_data()
                if attribution_data is not None and not attribution_data.empty:
                    # Bar chart showing contribution by factor
                    fig = px.bar(
                        attribution_data,
                        x="factor",
                        y="contribution",
                        color="contribution",
                        title="Performance Attribution by Factor",
                        color_continuous_scale="RdYlGn",
                        text="contribution"
                    )
                    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No performance attribution data available")
            except Exception as e:
                st.error(f"Error loading performance attribution data: {str(e)}")
            
            # Expected vs. actual execution quality
            st.write("### Execution Quality Analysis")
            try:
                execution_data = get_execution_quality_comparison()
                if execution_data is not None and not execution_data.empty:
                    # Create comparison chart
                    fig = go.Figure()
                    metrics = execution_data["metric"].unique()
                    
                    for metric in metrics:
                        metric_data = execution_data[execution_data["metric"] == metric]
                        fig.add_trace(go.Bar(
                            x=[metric],
                            y=[metric_data["expected"].values[0]],
                            name="Expected",
                            marker_color="blue"
                        ))
                        fig.add_trace(go.Bar(
                            x=[metric],
                            y=[metric_data["actual"].values[0]],
                            name="Actual",
                            marker_color="red"
                        ))
                    
                    fig.update_layout(
                        barmode="group",
                        title="Expected vs. Actual Execution Metrics"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No execution quality comparison data available")
            except Exception as e:
                st.error(f"Error loading execution quality data: {str(e)}")
            
            # Strategy adaptation points
            st.write("### Strategy Adaptation Points")
            try:
                adaptation_data = get_strategy_adaptation_data()
                if adaptation_data is not None and not adaptation_data.empty:
                    # Timeline of adaptation events
                    fig = px.timeline(
                        adaptation_data,
                        x_start="start_date",
                        x_end="end_date",
                        y="strategy",
                        color="event_type",
                        hover_name="description",
                        title="Strategy Adaptation Timeline"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed adaptation events
                    st.dataframe(adaptation_data[["timestamp", "strategy", "event_type", "description", "impact"]], 
                                use_container_width=True)
                else:
                    st.info("No strategy adaptation data available")
            except Exception as e:
                st.error(f"Error loading adaptation data: {str(e)}")

def trade_log_tab():
    """Trade Log tab - Detailed record of all trades"""
    st.header("Trade Log")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
    
    with col2:
        strategy_filter = st.multiselect(
            "Strategy",
            options=get_available_strategies(),
            default=[]
        )
    
    with col3:
        result_filter = st.multiselect(
            "Result",
            options=["Win", "Loss"],
            default=[]
        )
    
    # Get trades based on filters
    try:
        if st.session_state["connected"]:
            trades_df = get_filtered_trades(
                start_date=date_range[0] if len(date_range) > 0 else None,
                end_date=date_range[1] if len(date_range) > 1 else None,
                strategies=strategy_filter if strategy_filter else None,
                results=result_filter if result_filter else None
            )
            
            if trades_df is not None and not trades_df.empty:
                # Display trade table
                st.dataframe(
                    trades_df[[
                        "trade_id", "open_time", "close_time", "symbol", 
                        "direction", "entry_price", "exit_price", "pnl",
                        "strategy", "context"
                    ]],
                    use_container_width=True
                )
                
                # Summary statistics
                st.subheader("Trade Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", len(trades_df))
                
                with col2:
                    win_count = len(trades_df[trades_df["pnl"] > 0])
                    win_rate = (win_count / len(trades_df)) * 100 if len(trades_df) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.2f}%")
                
                with col3:
                    avg_pnl = trades_df["pnl"].mean()
                    st.metric("Average PnL", f"${avg_pnl:.2f}")
                
                with col4:
                    total_pnl = trades_df["pnl"].sum()
                    st.metric("Total PnL", f"${total_pnl:.2f}")
                
                # Average holding time
                if "open_time" in trades_df.columns and "close_time" in trades_df.columns:
                    trades_df["duration"] = trades_df["close_time"] - trades_df["open_time"]
                    avg_duration = trades_df["duration"].mean()
                    if pd.notna(avg_duration):
                        hours = avg_duration.total_seconds() / 3600
                        st.metric("Average Holding Time", f"{hours:.2f} hours")
            else:
                st.info("No trades found matching the selected filters")
    except Exception as e:
        st.error(f"Error loading trade data: {str(e)}")

def approval_audit_tab():
    """Approval/Audit tab - For switching between paper and live trading"""
    st.header("Trading Mode Control & Audit")
    
    # Current Mode Display
    st.subheader("Current Mode")
    mode_color = {
        "PAPER MODE": "blue",
        "PAUSED": "orange",
        "ACTIVE": "green"
    }.get(st.session_state["trading_mode"], "gray")
    st.markdown(f"<h3 style='color:{mode_color};'>{st.session_state['trading_mode']}</h3>", unsafe_allow_html=True)
    
    # Approval Controls
    st.subheader("Trading Mode Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Paper/Live toggle based on current status
        if st.session_state["trading_mode"] == "PAPER MODE":
            if st.button("📝 Request Live Approval", use_container_width=True):
                st.session_state["approval_status"] = "Awaiting Approval"
                st.success("Live trading approval requested. Please confirm in the Approval section.")
                
        elif st.session_state["trading_mode"] == "ACTIVE":
            if st.button("⏸️ Pause Live Trading", use_container_width=True):
                st.session_state["trading_mode"] = "PAUSED"
                st.session_state["approval_status"] = "Live Trading Paused"
                # Save this state change to MongoDB
                save_trading_mode_change("PAUSED")
                st.success("Live trading paused. System will not take new positions.")
                time.sleep(1)  # Brief pause to show the message
                st.rerun()  # Refresh the UI to reflect the new state
                
        elif st.session_state["trading_mode"] == "PAUSED":
            if st.button("▶️ Resume Live Trading", use_container_width=True):
                st.session_state["trading_mode"] = "ACTIVE"
                st.session_state["approval_status"] = "Live Trading"
                # Save this state change to MongoDB
                save_trading_mode_change("ACTIVE")
                st.success("Live trading resumed.")
                time.sleep(1)  # Brief pause to show the message
                st.rerun()  # Refresh the UI to reflect the new state
    
    with col2:
        # Return to paper trading (always available)
        if st.session_state["trading_mode"] != "PAPER MODE":
            if st.button("📊 Return to Paper Trading", use_container_width=True):
                st.session_state["trading_mode"] = "PAPER MODE"
                st.session_state["approval_status"] = "In Paper Trading"
                # Save this state change to MongoDB
                save_trading_mode_change("PAPER MODE")
                st.success("Returned to paper trading mode.")
                time.sleep(1)  # Brief pause to show the message
                st.rerun()  # Refresh the UI to reflect the new state
    
    # Approval Section (only shown when awaiting approval)
    if st.session_state["approval_status"] == "Awaiting Approval":
        st.subheader("Live Trading Approval")  
        st.warning("⚠️ You are about to approve live trading with real capital!")
        
        # Key Performance Indicators Check
        st.subheader("Performance Check")
        kpi_checks = check_kpi_requirements()
        kpi_table = []
        all_passed = True
        
        for kpi in kpi_checks:
            kpi_table.append({
                "Metric": kpi["name"],
                "Current Value": kpi["value"],
                "Requirement": kpi["requirement"],
                "Status": "✅ Pass" if kpi["passed"] else "❌ Fail"
            })
            if not kpi["passed"]:
                all_passed = False
        
        kpi_df = pd.DataFrame(kpi_table)
        st.dataframe(kpi_df, use_container_width=True)
        
        # Notes field
        notes = st.text_area("Audit Notes", "", help="Add any notes about this approval decision")
        
        # Approval buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve Live Trading", use_container_width=True, disabled=not all_passed):
                st.session_state["trading_mode"] = "ACTIVE"
                st.session_state["approval_status"] = "Live Trading"
                # Save approval to MongoDB with notes
                save_approval_event(approved=True, notes=notes)
                st.success("Live trading approved and activated!")
                time.sleep(1)  # Brief pause to show the message
                st.rerun()  # Refresh the UI to reflect the new state
        
        with col2:
            if st.button("❌ Deny Approval", use_container_width=True):
                st.session_state["approval_status"] = "In Paper Trading"
                # Save denial to MongoDB with notes
                save_approval_event(approved=False, notes=notes)
                st.error("Live trading request denied. Continuing in paper trading mode.")
                time.sleep(1)  # Brief pause to show the message
                st.rerun()  # Refresh the UI to reflect the new state
    
    # Audit History
    st.subheader("Approval Audit History")
    try:
        if st.session_state["connected"]:
            audit_history = get_approval_audit_history()
            if audit_history is not None and not audit_history.empty:
                st.dataframe(audit_history, use_container_width=True)
            else:
                st.info("No approval audit history available")
    except Exception as e:
        st.error(f"Error loading audit history: {str(e)}")

def main():
    """Main dashboard function - Command Bridge layout"""
    # Title with version
    st.title("BensBot Command Bridge") 
    st.caption("Institutional-Grade Trading System - v2.0")
    
    # Connection status - we need MongoDB connection for the dashboard to work
    if not st.session_state["connected"]:
        # Show connection form
        with st.form("mongodb_connection"):
            st.subheader("Connect to Database")
            st.text_input("MongoDB URI", value=st.session_state["mongodb_uri"], key="mongodb_uri_input")
            st.text_input("Database Name", value=st.session_state["mongodb_database"], key="mongodb_database_input")
            st.text_input("API Base URL", value=st.session_state["api_base_url"], key="api_base_url_input")
            
            submitted = st.form_submit_button("Connect")
            if submitted:
                st.session_state["mongodb_uri"] = st.session_state["mongodb_uri_input"]
                st.session_state["mongodb_database"] = st.session_state["mongodb_database_input"]
                st.session_state["api_base_url"] = st.session_state["api_base_url_input"]
                
                if connect_to_mongodb():
                    st.success("Connected to MongoDB!")
                    time.sleep(1)  # Brief pause
                    st.rerun()  # Refresh to show the dashboard
                else:
                    st.error("Failed to connect to MongoDB. Please check your connection details.")
        
        # Early return if not connected
        return
    
    # Display the tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Performance Metrics", "🔥 Trade Log", "⚙️ Approval/Audit"])
    
    with tab1:
        dashboard_home_tab()
    
    with tab2:
        performance_metrics_tab()
    
    with tab3:
        trade_log_tab()
    
    with tab4:
        approval_audit_tab()
        
    # Auto-refresh the dashboard
    if st.session_state["connected"] and "refresh_interval" in st.session_state:
        time.sleep(st.session_state["refresh_interval"])
        st.rerun()

# Helper functions for the approval system and dashboard metrics
def get_pnl_since(start_time):
    """Get the PnL since a specific time"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history(
                    start_time=start_time,
                    end_time=datetime.now()
                )
                
                if trades is not None and not trades.empty:
                    return trades["pnl"].sum()
                return 0.0
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                trades = list(db.trades.find({
                    "close_time": {"$gte": start_time}
                }))
                
                return sum(trade.get("pnl", 0) for trade in trades)
    except Exception as e:
        logger.error(f"Error getting PnL: {str(e)}")
        return 0.0

def get_total_pnl(mode="paper"):
    """Get the total PnL for a specific trading mode"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history()
                
                if trades is not None and not trades.empty:
                    mode_trades = trades[trades["mode"] == mode]
                    return mode_trades["pnl"].sum() if not mode_trades.empty else 0.0
                return 0.0
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                trades = list(db.trades.find({
                    "mode": mode
                }))
                
                return sum(trade.get("pnl", 0) for trade in trades)
    except Exception as e:
        logger.error(f"Error getting PnL: {str(e)}")
        return 0.0

def get_win_rate():
    """Get the overall win rate percentage"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history()
                
                if trades is not None and not trades.empty:
                    win_count = len(trades[trades["pnl"] > 0])
                    return (win_count / len(trades)) * 100 if len(trades) > 0 else 0.0
                return 0.0
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                all_trades = list(db.trades.find())
                
                if all_trades:
                    win_count = sum(1 for trade in all_trades if trade.get("pnl", 0) > 0)
                    return (win_count / len(all_trades)) * 100
                return 0.0
    except Exception as e:
        logger.error(f"Error calculating win rate: {str(e)}")
        return 0.0

def get_average_trade_gain():
    """Get the average percentage gain/loss per trade"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history()
                
                if trades is not None and not trades.empty:
                    # Calculate percentage gain
                    trades["pct_gain"] = trades.apply(
                        lambda x: (x["pnl"] / x["position_size"]) * 100 if x["position_size"] > 0 else 0, 
                        axis=1
                    )
                    return trades["pct_gain"].mean()
                return 0.0
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                all_trades = list(db.trades.find())
                
                if all_trades:
                    pct_gains = []
                    for trade in all_trades:
                        if trade.get("position_size", 0) > 0:
                            pct_gain = (trade.get("pnl", 0) / trade.get("position_size", 1)) * 100
                            pct_gains.append(pct_gain)
                    
                    return sum(pct_gains) / len(pct_gains) if pct_gains else 0.0
                return 0.0
    except Exception as e:
        logger.error(f"Error calculating average gain: {str(e)}")
        return 0.0

def get_sharpe_ratio():
    """Calculate the Sharpe ratio"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                performance = persistence.get_performance_history(metric_type="daily_return")
                
                if performance is not None and not performance.empty:
                    daily_returns = performance["value"]
                    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
                return 0.0
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                daily_returns = list(db.performance.find({"metric_type": "daily_return"}))
                
                if daily_returns:
                    values = [item.get("value", 0) for item in daily_returns]
                    mean_return = sum(values) / len(values)
                    std_return = np.std(values) if len(values) > 1 else 1e-6
                    return (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
                return 0.0
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0.0

def get_current_drawdown():
    """Get the current drawdown percentage"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                capital_state = persistence.load_strategy_state("capital_manager")
                
                if capital_state and "current_drawdown" in capital_state:
                    return capital_state["current_drawdown"]
                return 0.0
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                capital_state = db.strategy_states.find_one({"strategy_id": "capital_manager"})
                
                if capital_state and "data" in capital_state and "current_drawdown" in capital_state["data"]:
                    return capital_state["data"]["current_drawdown"] * 100
                return 0.0
    except Exception as e:
        logger.error(f"Error getting drawdown: {str(e)}")
        return 0.0

def get_equity_curve_data():
    """Get the equity curve data for plotting"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                performance = persistence.get_performance_history(metric_type="equity")
                return performance
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                equity_data = list(db.performance.find({"metric_type": "equity"}))
                
                if equity_data:
                    return pd.DataFrame(equity_data)
                return None
    except Exception as e:
        logger.error(f"Error getting equity curve data: {str(e)}")
        return None

def get_win_loss_distribution():
    """Get the win/loss distribution for pie chart"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history()
                
                if trades is not None and not trades.empty:
                    wins = len(trades[trades["pnl"] > 0])
                    losses = len(trades[trades["pnl"] <= 0])
                    return {"wins": wins, "losses": losses}
                return {"wins": 0, "losses": 0}
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                all_trades = list(db.trades.find())
                
                wins = sum(1 for trade in all_trades if trade.get("pnl", 0) > 0)
                losses = len(all_trades) - wins
                return {"wins": wins, "losses": losses}
    except Exception as e:
        logger.error(f"Error getting win/loss distribution: {str(e)}")
        return {"wins": 0, "losses": 0}

def get_strategy_allocation():
    """Get the strategy allocation data for pie chart"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                capital_state = persistence.load_strategy_state("capital_manager")
                
                if capital_state and "strategy_allocations" in capital_state:
                    allocations = capital_state["strategy_allocations"]
                    return pd.DataFrame({
                        "strategy": list(allocations.keys()),
                        "allocation": list(allocations.values())
                    })
                return None
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                capital_state = db.strategy_states.find_one({"strategy_id": "capital_manager"})
                
                if capital_state and "data" in capital_state and "strategy_allocations" in capital_state["data"]:
                    allocations = capital_state["data"]["strategy_allocations"]
                    return pd.DataFrame({
                        "strategy": list(allocations.keys()),
                        "allocation": list(allocations.values())
                    })
                return None
    except Exception as e:
        logger.error(f"Error getting strategy allocation data: {str(e)}")
        return None

def get_available_strategies():
    """Get list of all available strategies"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history()
                
                if trades is not None and not trades.empty:
                    return trades["strategy"].unique().tolist()
                return []
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                strategies = db.trades.distinct("strategy")
                return strategies
    except Exception as e:
        logger.error(f"Error getting available strategies: {str(e)}")
        return []

def get_filtered_trades(start_date=None, end_date=None, strategies=None, results=None):
    """Get trades filtered by criteria"""
    try:
        if st.session_state["connected"]:
            # Build filter
            query = {}
            if start_date:
                if "start_time" not in query:
                    query["close_time"] = {}
                query["close_time"]["$gte"] = datetime.combine(start_date, datetime.min.time())
            
            if end_date:
                if "close_time" not in query:
                    query["close_time"] = {}
                query["close_time"]["$lte"] = datetime.combine(end_date, datetime.max.time())
            
            if strategies:
                query["strategy"] = {"$in": strategies}
            
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                trades = persistence.get_trades_history(**query)
                
                # Filter by result
                if results and trades is not None and not trades.empty:
                    if "Win" in results and "Loss" in results:
                        # Both selected, no filtering needed
                        pass
                    elif "Win" in results:
                        trades = trades[trades["pnl"] > 0]
                    elif "Loss" in results:
                        trades = trades[trades["pnl"] <= 0]
                
                return trades
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                trades = list(db.trades.find(query))
                
                # Convert to DataFrame
                if trades:
                    trades_df = pd.DataFrame(trades)
                    
                    # Filter by result
                    if results:
                        if "Win" in results and "Loss" in results:
                            # Both selected, no filtering needed
                            pass
                        elif "Win" in results:
                            trades_df = trades_df[trades_df["pnl"] > 0]
                        elif "Loss" in results:
                            trades_df = trades_df[trades_df["pnl"] <= 0]
                    
                    return trades_df
                return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting filtered trades: {str(e)}")
        return pd.DataFrame()

def save_trading_mode_change(new_mode):
    """Save trading mode change to MongoDB"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                persistence.log_system_event(
                    level="INFO",
                    message=f"Trading mode changed to: {new_mode}",
                    component="ApprovalGate"
                )
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                db.system_logs.insert_one({
                    "timestamp": datetime.now(),
                    "level": "INFO",
                    "message": f"Trading mode changed to: {new_mode}",
                    "component": "ApprovalGate"
                })
            return True
    except Exception as e:
        logger.error(f"Error saving trading mode change: {str(e)}")
        return False

def check_kpi_requirements():
    """Check if performance KPIs meet requirements for live trading approval"""
    kpi_checks = [
        {
            "name": "Win Rate",
            "value": f"{get_win_rate():.2f}%",
            "requirement": ">= 55%",
            "passed": get_win_rate() >= 55.0
        },
        {
            "name": "Sharpe Ratio",
            "value": f"{get_sharpe_ratio():.2f}",
            "requirement": ">= 1.0",
            "passed": get_sharpe_ratio() >= 1.0
        },
        {
            "name": "Maximum Drawdown",
            "value": f"{get_current_drawdown():.2f}%",
            "requirement": "<= 15%",
            "passed": get_current_drawdown() <= 15.0
        },
        {
            "name": "Minimum Paper Trades",
            "value": f"{len(get_filtered_trades())}",
            "requirement": ">= 30 trades",
            "passed": len(get_filtered_trades()) >= 30
        }
    ]
    return kpi_checks

def save_approval_event(approved, notes):
    """Save approval/denial event to MongoDB"""
    try:
        if st.session_state["connected"]:
            event_data = {
                "timestamp": datetime.now(),
                "action": "Approval" if approved else "Denial",
                "notes": notes,
                "previous_mode": "PAPER MODE",
                "new_mode": "ACTIVE" if approved else "PAPER MODE"
            }
            
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                persistence.log_system_event(
                    level="INFO",
                    message=f"Trading mode {event_data['action']}: {event_data['notes']}",
                    component="ApprovalGate"
                )
                
                # Also save to dedicated collection if exists
                client = persistence.client
                db = client[persistence.database_name]
                if "approval_audit" not in db.list_collection_names():
                    db.create_collection("approval_audit")
                db.approval_audit.insert_one(event_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                # Log to system logs
                db.system_logs.insert_one({
                    "timestamp": datetime.now(),
                    "level": "INFO",
                    "message": f"Trading mode {event_data['action']}: {event_data['notes']}",
                    "component": "ApprovalGate"
                })
                
                # Also save to dedicated collection
                if "approval_audit" not in db.list_collection_names():
                    db.create_collection("approval_audit")
                db.approval_audit.insert_one(event_data)
            return True
    except Exception as e:
        logger.error(f"Error saving approval event: {str(e)}")
        return False

def get_approval_audit_history():
    """Get approval audit history from MongoDB"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                client = persistence.client
                db = client[persistence.database_name]
                
                if "approval_audit" in db.list_collection_names():
                    audit_logs = list(db.approval_audit.find().sort("timestamp", -1))
                    if audit_logs:
                        return pd.DataFrame(audit_logs)
                return None
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                if "approval_audit" in db.list_collection_names():
                    audit_logs = list(db.approval_audit.find().sort("timestamp", -1))
                    if audit_logs:
                        return pd.DataFrame(audit_logs)
                return None
    except Exception as e:
        logger.error(f"Error getting approval audit history: {str(e)}")
        return None

# Strategy Intelligence helper functions
def get_market_analysis_data():
    """Get market analysis data that led to asset class selection"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                strategy_state = persistence.load_strategy_state("market_analysis")
                
                if strategy_state and "asset_classes" in strategy_state:
                    asset_classes = strategy_state["asset_classes"]
                    return pd.DataFrame(asset_classes)
                    
                # If no data in database, generate mock data for demonstration
                mock_data = [
                    {"asset_class": "forex", "opportunity_score": 82, "selection_reason": "High volatility across major pairs with current market regime"}, 
                    {"asset_class": "crypto", "opportunity_score": 68, "selection_reason": "Increasing institutional adoption creating new trading opportunities"},
                    {"asset_class": "stocks", "opportunity_score": 59, "selection_reason": "Earnings season shows potential for mid-cap opportunities"},
                    {"asset_class": "commodities", "opportunity_score": 45, "selection_reason": "Oil market consolidation limits short-term trading potential"},
                    {"asset_class": "bonds", "opportunity_score": 31, "selection_reason": "Low volatility environment with minimal short-term opportunities"}
                ]
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                market_state = db.strategy_states.find_one({"strategy_id": "market_analysis"})
                
                if market_state and "data" in market_state and "asset_classes" in market_state["data"]:
                    asset_classes = market_state["data"]["asset_classes"]
                    return pd.DataFrame(asset_classes)
                    
                # If no data in database, generate mock data for demonstration
                mock_data = [
                    {"asset_class": "forex", "opportunity_score": 82, "selection_reason": "High volatility across major pairs with current market regime"}, 
                    {"asset_class": "crypto", "opportunity_score": 68, "selection_reason": "Increasing institutional adoption creating new trading opportunities"},
                    {"asset_class": "stocks", "opportunity_score": 59, "selection_reason": "Earnings season shows potential for mid-cap opportunities"},
                    {"asset_class": "commodities", "opportunity_score": 45, "selection_reason": "Oil market consolidation limits short-term trading potential"},
                    {"asset_class": "bonds", "opportunity_score": 31, "selection_reason": "Low volatility environment with minimal short-term opportunities"}
                ]
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting market analysis data: {str(e)}")
        return None

def get_asset_correlation_data():
    """Get correlation data between selected assets"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                correlation_state = persistence.load_strategy_state("correlation_matrix")
                
                if correlation_state and "matrix" in correlation_state:
                    return correlation_state["matrix"]
                    
                # If no data in database, generate mock data for demonstration
                assets = ["EUR/USD", "BTC/USD", "ETH/USD", "USD/JPY", "GBP/USD"]
                mock_matrix = pd.DataFrame(np.random.uniform(-1, 1, size=(len(assets), len(assets))), 
                                         columns=assets, index=assets)
                # Make diagonal 1.0
                for i in range(len(assets)):
                    mock_matrix.iloc[i, i] = 1.0
                # Make symmetric
                mock_matrix = (mock_matrix + mock_matrix.T) / 2
                return mock_matrix
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                correlation_state = db.strategy_states.find_one({"strategy_id": "correlation_matrix"})
                
                if correlation_state and "data" in correlation_state and "matrix" in correlation_state["data"]:
                    matrix_data = correlation_state["data"]["matrix"]
                    return pd.DataFrame(matrix_data)
                    
                # If no data in database, generate mock data for demonstration
                assets = ["EUR/USD", "BTC/USD", "ETH/USD", "USD/JPY", "GBP/USD"]
                mock_matrix = pd.DataFrame(np.random.uniform(-1, 1, size=(len(assets), len(assets))), 
                                         columns=assets, index=assets)
                # Make diagonal 1.0
                for i in range(len(assets)):
                    mock_matrix.iloc[i, i] = 1.0
                # Make symmetric
                mock_matrix = (mock_matrix + mock_matrix.T) / 2
                return mock_matrix
    except Exception as e:
        logger.error(f"Error getting asset correlation data: {str(e)}")
        return None

def get_symbol_ranking_data():
    """Get symbol ranking data for symbol selection"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                symbol_ranking = persistence.load_strategy_state("symbol_selection")
                
                if symbol_ranking and "rankings" in symbol_ranking:
                    return pd.DataFrame(symbol_ranking["rankings"])
                    
                # If no data in database, generate mock data for demonstration
                mock_data = [
                    {"symbol": "EUR/USD", "liquidity": 95, "volatility": 72, "spread": 88, "trend_strength": 65, "regime_fit": 80, "total_score": 82, "rank": 1},
                    {"symbol": "GBP/USD", "liquidity": 92, "volatility": 78, "spread": 85, "trend_strength": 70, "regime_fit": 75, "total_score": 79, "rank": 2},
                    {"symbol": "USD/JPY", "liquidity": 90, "volatility": 65, "spread": 87, "trend_strength": 62, "regime_fit": 72, "total_score": 75, "rank": 3},
                    {"symbol": "AUD/USD", "liquidity": 85, "volatility": 80, "spread": 83, "trend_strength": 58, "regime_fit": 68, "total_score": 72, "rank": 4},
                    {"symbol": "USD/CAD", "liquidity": 82, "volatility": 68, "spread": 80, "trend_strength": 60, "regime_fit": 65, "total_score": 70, "rank": 5},
                    {"symbol": "EUR/JPY", "liquidity": 80, "volatility": 85, "spread": 78, "trend_strength": 75, "regime_fit": 58, "total_score": 69, "rank": 6},
                    {"symbol": "USD/CHF", "liquidity": 78, "volatility": 60, "spread": 76, "trend_strength": 55, "regime_fit": 60, "total_score": 65, "rank": 7},
                    {"symbol": "NZD/USD", "liquidity": 72, "volatility": 75, "spread": 70, "trend_strength": 50, "regime_fit": 55, "total_score": 62, "rank": 8}
                ]
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                symbol_ranking = db.strategy_states.find_one({"strategy_id": "symbol_selection"})
                
                if symbol_ranking and "data" in symbol_ranking and "rankings" in symbol_ranking["data"]:
                    return pd.DataFrame(symbol_ranking["data"]["rankings"])
                    
                # If no data in database, generate mock data for demonstration
                mock_data = [
                    {"symbol": "EUR/USD", "liquidity": 95, "volatility": 72, "spread": 88, "trend_strength": 65, "regime_fit": 80, "total_score": 82, "rank": 1},
                    {"symbol": "GBP/USD", "liquidity": 92, "volatility": 78, "spread": 85, "trend_strength": 70, "regime_fit": 75, "total_score": 79, "rank": 2},
                    {"symbol": "USD/JPY", "liquidity": 90, "volatility": 65, "spread": 87, "trend_strength": 62, "regime_fit": 72, "total_score": 75, "rank": 3},
                    {"symbol": "AUD/USD", "liquidity": 85, "volatility": 80, "spread": 83, "trend_strength": 58, "regime_fit": 68, "total_score": 72, "rank": 4},
                    {"symbol": "USD/CAD", "liquidity": 82, "volatility": 68, "spread": 80, "trend_strength": 60, "regime_fit": 65, "total_score": 70, "rank": 5},
                    {"symbol": "EUR/JPY", "liquidity": 80, "volatility": 85, "spread": 78, "trend_strength": 75, "regime_fit": 58, "total_score": 69, "rank": 6},
                    {"symbol": "USD/CHF", "liquidity": 78, "volatility": 60, "spread": 76, "trend_strength": 55, "regime_fit": 60, "total_score": 65, "rank": 7},
                    {"symbol": "NZD/USD", "liquidity": 72, "volatility": 75, "spread": 70, "trend_strength": 50, "regime_fit": 55, "total_score": 62, "rank": 8}
                ]
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting symbol ranking data: {str(e)}")
        return None

def get_symbol_historical_performance():
    """Get historical performance data for symbols in similar market conditions"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                historical_perf = persistence.load_strategy_state("symbol_historical_performance")
                
                if historical_perf and "performance" in historical_perf:
                    return pd.DataFrame(historical_perf["performance"])
                    
                # If no data in database, generate mock data for demonstration
                market_regimes = ["trending", "ranging", "volatile", "low_volatility"]
                symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"] 
                
                mock_data = []
                for symbol in symbols:
                    for regime in market_regimes:
                        # Generate performance between -5 and 15
                        perf = (np.random.random() * 20) - 5
                        # Adjust based on common patterns
                        if regime == "trending" and symbol in ["EUR/USD", "GBP/USD"]:
                            perf += 5  # These perform better in trending markets
                        if regime == "ranging" and symbol in ["USD/JPY", "USD/CAD"]:
                            perf += 3  # These perform better in ranging markets
                        
                        mock_data.append({
                            "symbol": symbol,
                            "market_regime": regime,
                            "performance": round(perf, 2)
                        })
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                historical_perf = db.strategy_states.find_one({"strategy_id": "symbol_historical_performance"})
                
                if historical_perf and "data" in historical_perf and "performance" in historical_perf["data"]:
                    return pd.DataFrame(historical_perf["data"]["performance"])
                    
                # If no data in database, generate mock data for demonstration
                market_regimes = ["trending", "ranging", "volatile", "low_volatility"]
                symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"] 
                
                mock_data = []
                for symbol in symbols:
                    for regime in market_regimes:
                        # Generate performance between -5 and 15
                        perf = (np.random.random() * 20) - 5
                        # Adjust based on common patterns
                        if regime == "trending" and symbol in ["EUR/USD", "GBP/USD"]:
                            perf += 5  # These perform better in trending markets
                        if regime == "ranging" and symbol in ["USD/JPY", "USD/CAD"]:
                            perf += 3  # These perform better in ranging markets
                        
                        mock_data.append({
                            "symbol": symbol,
                            "market_regime": regime,
                            "performance": round(perf, 2)
                        })
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting symbol historical performance data: {str(e)}")
        return None

def get_market_regime_data():
    """Get current market regime detection data"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                regime_data = persistence.load_strategy_state("market_regime_detector")
                
                if regime_data and "current_regime" in regime_data:
                    return regime_data
                    
                # If no data in database, generate mock data for demonstration
                regimes = ["trending", "ranging", "volatile", "low_volatility"]
                current_regime = np.random.choice(regimes)
                confidence = round(np.random.uniform(0.70, 0.95), 2)
                
                # Generate history of regime changes
                now = pd.Timestamp.now()
                history = []
                for i in range(5):
                    regime = np.random.choice(regimes)
                    # Each regime lasted between 3 and 15 days
                    duration_days = np.random.randint(3, 15)
                    end_date = now - pd.Timedelta(days=i*duration_days)
                    start_date = end_date - pd.Timedelta(days=duration_days)
                    
                    history.append({
                        "timestamp": end_date,
                        "regime": regime,
                        "confidence": round(np.random.uniform(0.6, 0.95), 2),
                        "duration_days": duration_days,
                        "trigger": np.random.choice(["volatility_spike", "trend_reversal", "consolidation", "breakout"])
                    })
                
                mock_data = {
                    "current_regime": current_regime,
                    "confidence": confidence,
                    "detected_at": str(now),
                    "history": history
                }
                return mock_data
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                regime_data = db.strategy_states.find_one({"strategy_id": "market_regime_detector"})
                
                if regime_data and "data" in regime_data and "current_regime" in regime_data["data"]:
                    return regime_data["data"]
                    
                # If no data in database, generate mock data for demonstration
                regimes = ["trending", "ranging", "volatile", "low_volatility"]
                current_regime = np.random.choice(regimes)
                confidence = round(np.random.uniform(0.70, 0.95), 2)
                
                # Generate history of regime changes
                now = pd.Timestamp.now()
                history = []
                for i in range(5):
                    regime = np.random.choice(regimes)
                    # Each regime lasted between 3 and 15 days
                    duration_days = np.random.randint(3, 15)
                    end_date = now - pd.Timedelta(days=i*duration_days)
                    start_date = end_date - pd.Timedelta(days=duration_days)
                    
                    history.append({
                        "timestamp": end_date,
                        "regime": regime,
                        "confidence": round(np.random.uniform(0.6, 0.95), 2),
                        "duration_days": duration_days,
                        "trigger": np.random.choice(["volatility_spike", "trend_reversal", "consolidation", "breakout"])
                    })
                
                mock_data = {
                    "current_regime": current_regime,
                    "confidence": confidence,
                    "detected_at": str(now),
                    "history": history
                }
                return mock_data
    except Exception as e:
        logger.error(f"Error getting market regime data: {str(e)}")
        return None

def get_strategy_compatibility_data():
    """Get strategy compatibility matrix with market regimes"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                compatibility = persistence.load_strategy_state("strategy_compatibility")
                
                if compatibility and "matrix" in compatibility:
                    return pd.DataFrame(compatibility["matrix"])
                    
                # If no data in database, generate mock data for demonstration
                strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
                regimes = ["trending", "ranging", "volatile", "low_volatility"]
                
                # Create a compatibility matrix with some sensible values
                mock_data = pd.DataFrame(index=strategies, columns=regimes)
                
                # Fill with realistic compatibility scores (0-100)
                # Trend following works well in trending markets
                mock_data.loc["TrendFollowing", "trending"] = 95
                mock_data.loc["TrendFollowing", "ranging"] = 30
                mock_data.loc["TrendFollowing", "volatile"] = 45
                mock_data.loc["TrendFollowing", "low_volatility"] = 50
                
                # Mean reversion works well in ranging markets
                mock_data.loc["MeanReversion", "trending"] = 35
                mock_data.loc["MeanReversion", "ranging"] = 90
                mock_data.loc["MeanReversion", "volatile"] = 40
                mock_data.loc["MeanReversion", "low_volatility"] = 60
                
                # Breakout works well in volatile markets
                mock_data.loc["Breakout", "trending"] = 60
                mock_data.loc["Breakout", "ranging"] = 40
                mock_data.loc["Breakout", "volatile"] = 88
                mock_data.loc["Breakout", "low_volatility"] = 20
                
                # Momentum works well in trending markets 
                mock_data.loc["Momentum", "trending"] = 90
                mock_data.loc["Momentum", "ranging"] = 35
                mock_data.loc["Momentum", "volatile"] = 60
                mock_data.loc["Momentum", "low_volatility"] = 40
                
                # Scalping works across the board but best in volatile markets
                mock_data.loc["Scalping", "trending"] = 65
                mock_data.loc["Scalping", "ranging"] = 70
                mock_data.loc["Scalping", "volatile"] = 85
                mock_data.loc["Scalping", "low_volatility"] = 55
                
                return mock_data
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                compatibility = db.strategy_states.find_one({"strategy_id": "strategy_compatibility"})
                
                if compatibility and "data" in compatibility and "matrix" in compatibility["data"]:
                    matrix_data = compatibility["data"]["matrix"]
                    return pd.DataFrame(matrix_data)
                    
                # If no data in database, generate mock data for demonstration
                strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
                regimes = ["trending", "ranging", "volatile", "low_volatility"]
                
                # Create a compatibility matrix with some sensible values
                mock_data = pd.DataFrame(index=strategies, columns=regimes)
                
                # Fill with realistic compatibility scores (0-100)
                # Trend following works well in trending markets
                mock_data.loc["TrendFollowing", "trending"] = 95
                mock_data.loc["TrendFollowing", "ranging"] = 30
                mock_data.loc["TrendFollowing", "volatile"] = 45
                mock_data.loc["TrendFollowing", "low_volatility"] = 50
                
                # Mean reversion works well in ranging markets
                mock_data.loc["MeanReversion", "trending"] = 35
                mock_data.loc["MeanReversion", "ranging"] = 90
                mock_data.loc["MeanReversion", "volatile"] = 40
                mock_data.loc["MeanReversion", "low_volatility"] = 60
                
                # Breakout works well in volatile markets
                mock_data.loc["Breakout", "trending"] = 60
                mock_data.loc["Breakout", "ranging"] = 40
                mock_data.loc["Breakout", "volatile"] = 88
                mock_data.loc["Breakout", "low_volatility"] = 20
                
                # Momentum works well in trending markets 
                mock_data.loc["Momentum", "trending"] = 90
                mock_data.loc["Momentum", "ranging"] = 35
                mock_data.loc["Momentum", "volatile"] = 60
                mock_data.loc["Momentum", "low_volatility"] = 40
                
                # Scalping works across the board but best in volatile markets
                mock_data.loc["Scalping", "trending"] = 65
                mock_data.loc["Scalping", "ranging"] = 70
                mock_data.loc["Scalping", "volatile"] = 85
                mock_data.loc["Scalping", "low_volatility"] = 55
                
                return mock_data
    except Exception as e:
        logger.error(f"Error getting strategy compatibility data: {str(e)}")
        return None

def get_strategy_performance_benchmarks():
    """Get strategy performance benchmarks in similar market regimes"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                benchmark_data = persistence.load_strategy_state("strategy_benchmarks")
                
                if benchmark_data and "benchmarks" in benchmark_data:
                    return pd.DataFrame(benchmark_data["benchmarks"])
                    
                # If no data in database, generate mock data for demonstration
                # Get current market regime
                regime_data = get_market_regime_data()
                current_regime = "trending"  # default
                if regime_data and "current_regime" in regime_data:
                    current_regime = regime_data["current_regime"]
                
                strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
                metrics = ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "avg_trade"]
                
                mock_data = []
                for strategy in strategies:
                    # Base performance
                    win_rate = np.random.uniform(0.35, 0.65)
                    profit_factor = np.random.uniform(1.0, 2.5)
                    sharpe = np.random.uniform(0.5, 2.5)
                    max_dd = np.random.uniform(0.05, 0.25)
                    avg_trade = np.random.uniform(-0.5, 1.5)
                    
                    # Adjust based on strategy-regime fit
                    if (strategy == "TrendFollowing" and current_regime == "trending") or \
                       (strategy == "MeanReversion" and current_regime == "ranging") or \
                       (strategy == "Breakout" and current_regime == "volatile"):
                        # Boost performance for strategies that match current regime
                        win_rate += 0.15
                        profit_factor += 0.8
                        sharpe += 0.7
                        max_dd -= 0.05
                        avg_trade += 0.7
                    
                    mock_data.append({
                        "strategy": strategy,
                        "regime": current_regime,
                        "win_rate": round(win_rate * 100, 1),
                        "profit_factor": round(profit_factor, 2),
                        "sharpe_ratio": round(sharpe, 2),
                        "max_drawdown": round(max_dd * 100, 1),
                        "avg_trade_pct": round(avg_trade, 2),
                        "sample_size": np.random.randint(50, 500)
                    })
                
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                benchmark_data = db.strategy_states.find_one({"strategy_id": "strategy_benchmarks"})
                
                if benchmark_data and "data" in benchmark_data and "benchmarks" in benchmark_data["data"]:
                    return pd.DataFrame(benchmark_data["data"]["benchmarks"])
                    
                # If no data in database, generate mock data for demonstration
                # Get current market regime
                regime_data = get_market_regime_data()
                current_regime = "trending"  # default
                if regime_data and "current_regime" in regime_data:
                    current_regime = regime_data["current_regime"]
                
                strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
                metrics = ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "avg_trade"]
                
                mock_data = []
                for strategy in strategies:
                    # Base performance
                    win_rate = np.random.uniform(0.35, 0.65)
                    profit_factor = np.random.uniform(1.0, 2.5)
                    sharpe = np.random.uniform(0.5, 2.5)
                    max_dd = np.random.uniform(0.05, 0.25)
                    avg_trade = np.random.uniform(-0.5, 1.5)
                    
                    # Adjust based on strategy-regime fit
                    if (strategy == "TrendFollowing" and current_regime == "trending") or \
                       (strategy == "MeanReversion" and current_regime == "ranging") or \
                       (strategy == "Breakout" and current_regime == "volatile"):
                        # Boost performance for strategies that match current regime
                        win_rate += 0.15
                        profit_factor += 0.8
                        sharpe += 0.7
                        max_dd -= 0.05
                        avg_trade += 0.7
                    
                    mock_data.append({
                        "strategy": strategy,
                        "regime": current_regime,
                        "win_rate": round(win_rate * 100, 1),
                        "profit_factor": round(profit_factor, 2),
                        "sharpe_ratio": round(sharpe, 2),
                        "max_drawdown": round(max_dd * 100, 1),
                        "avg_trade_pct": round(avg_trade, 2),
                        "sample_size": np.random.randint(50, 500)
                    })
                
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting strategy benchmark data: {str(e)}")
        return None

def get_performance_attribution_data():
    """Get performance attribution data showing what factors contributed to performance"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                attribution = persistence.load_strategy_state("performance_attribution")
                
                if attribution and "factors" in attribution:
                    return pd.DataFrame(attribution["factors"])
                    
                # If no data in database, generate mock data for demonstration
                mock_data = [
                    {"factor": "Market Regime Detection", "contribution": 42.5},
                    {"factor": "Asset Selection", "contribution": 18.7},
                    {"factor": "Position Sizing", "contribution": 15.3},
                    {"factor": "Entry Timing", "contribution": 12.2},
                    {"factor": "Exit Timing", "contribution": 8.6},
                    {"factor": "Execution Quality", "contribution": -2.8},
                    {"factor": "Fees & Slippage", "contribution": -5.5},
                    {"factor": "Other Factors", "contribution": 11.0}
                ]
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                attribution = db.strategy_states.find_one({"strategy_id": "performance_attribution"})
                
                if attribution and "data" in attribution and "factors" in attribution["data"]:
                    return pd.DataFrame(attribution["data"]["factors"])
                    
                # If no data in database, generate mock data for demonstration
                mock_data = [
                    {"factor": "Market Regime Detection", "contribution": 42.5},
                    {"factor": "Asset Selection", "contribution": 18.7},
                    {"factor": "Position Sizing", "contribution": 15.3},
                    {"factor": "Entry Timing", "contribution": 12.2},
                    {"factor": "Exit Timing", "contribution": 8.6},
                    {"factor": "Execution Quality", "contribution": -2.8},
                    {"factor": "Fees & Slippage", "contribution": -5.5},
                    {"factor": "Other Factors", "contribution": 11.0}
                ]
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting performance attribution data: {str(e)}")
        return None

def get_execution_quality_comparison():
    """Get comparison between expected and actual execution quality metrics"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                execution = persistence.load_strategy_state("execution_quality")
                
                if execution and "metrics" in execution:
                    return pd.DataFrame(execution["metrics"])
                    
                # If no data in database, generate mock data for demonstration
                metrics = ["Slippage (pips)", "Spread (pips)", "Latency (ms)", "Rejection Rate (%)", "Fill Ratio (%)"]
                
                mock_data = []
                for metric in metrics:
                    # Base expected value
                    if metric == "Slippage (pips)":
                        expected = 1.2
                        actual = round(expected * (1 + np.random.uniform(-0.3, 0.5)), 1)
                    elif metric == "Spread (pips)":
                        expected = 1.5
                        actual = round(expected * (1 + np.random.uniform(-0.1, 0.2)), 1)
                    elif metric == "Latency (ms)":
                        expected = 250
                        actual = int(expected * (1 + np.random.uniform(-0.1, 0.3)))
                    elif metric == "Rejection Rate (%)":
                        expected = 0.8
                        actual = round(expected * (1 + np.random.uniform(0, 0.5)), 1)
                    elif metric == "Fill Ratio (%)":
                        expected = 99.2
                        actual = round(min(100, expected * (1 + np.random.uniform(-0.02, 0))), 1)
                    
                    mock_data.append({
                        "metric": metric,
                        "expected": expected,
                        "actual": actual,
                        "difference": round(((actual - expected) / expected) * 100, 1) if expected != 0 else 0
                    })
                
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                execution = db.strategy_states.find_one({"strategy_id": "execution_quality"})
                
                if execution and "data" in execution and "metrics" in execution["data"]:
                    return pd.DataFrame(execution["data"]["metrics"])
                    
                # If no data in database, generate mock data for demonstration
                metrics = ["Slippage (pips)", "Spread (pips)", "Latency (ms)", "Rejection Rate (%)", "Fill Ratio (%)"]
                
                mock_data = []
                for metric in metrics:
                    # Base expected value
                    if metric == "Slippage (pips)":
                        expected = 1.2
                        actual = round(expected * (1 + np.random.uniform(-0.3, 0.5)), 1)
                    elif metric == "Spread (pips)":
                        expected = 1.5
                        actual = round(expected * (1 + np.random.uniform(-0.1, 0.2)), 1)
                    elif metric == "Latency (ms)":
                        expected = 250
                        actual = int(expected * (1 + np.random.uniform(-0.1, 0.3)))
                    elif metric == "Rejection Rate (%)":
                        expected = 0.8
                        actual = round(expected * (1 + np.random.uniform(0, 0.5)), 1)
                    elif metric == "Fill Ratio (%)":
                        expected = 99.2
                        actual = round(min(100, expected * (1 + np.random.uniform(-0.02, 0))), 1)
                    
                    mock_data.append({
                        "metric": metric,
                        "expected": expected,
                        "actual": actual,
                        "difference": round(((actual - expected) / expected) * 100, 1) if expected != 0 else 0
                    })
                
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting execution quality comparison data: {str(e)}")
        return None

def get_strategy_adaptation_data():
    """Get data on strategy adaptation points"""
    try:
        if st.session_state["connected"]:
            if COMPONENTS_IMPORTED and st.session_state["persistence"]:
                # Use persistence manager
                persistence = st.session_state["persistence"]
                adaptation = persistence.load_strategy_state("strategy_adaptation")
                
                if adaptation and "events" in adaptation:
                    return pd.DataFrame(adaptation["events"])
                    
                # If no data in database, generate mock data for demonstration
                now = pd.Timestamp.now()
                strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
                event_types = ["Parameter Adjustment", "Regime Change Response", "Risk Adjustment", "New Feature", "Filter Added"]
                impacts = ["Positive", "Neutral", "Negative"]
                
                mock_data = []
                for i in range(15):
                    days_ago = np.random.randint(1, 60)
                    timestamp = now - pd.Timedelta(days=days_ago)
                    strategy = np.random.choice(strategies)
                    event_type = np.random.choice(event_types)
                    
                    # Create realistic descriptions
                    if event_type == "Parameter Adjustment":
                        desc = f"Adjusted {strategy} lookback period from {np.random.randint(10, 30)} to {np.random.randint(10, 30)} days"
                    elif event_type == "Regime Change Response":
                        desc = f"Switched {strategy} to {np.random.choice(['conservative', 'aggressive', 'neutral'])} mode due to regime change"
                    elif event_type == "Risk Adjustment":
                        desc = f"Modified {strategy} risk per trade from {np.random.uniform(0.5, 2.0):.1f}% to {np.random.uniform(0.5, 2.0):.1f}%"
                    elif event_type == "New Feature":
                        desc = f"Added {np.random.choice(['volatility filter', 'volume confirmation', 'news filter', 'correlation check'])} to {strategy}"
                    elif event_type == "Filter Added":
                        desc = f"Implemented {np.random.choice(['ADX filter', 'RSI filter', 'Bollinger filter', 'MA crossover filter'])} to {strategy}"
                    
                    impact = np.random.choice(impacts, p=[0.6, 0.3, 0.1])  # Bias toward positive outcomes
                    impact_value = np.random.uniform(-5, 15) if impact == "Positive" else \
                                  np.random.uniform(-2, 2) if impact == "Neutral" else \
                                  np.random.uniform(-12, -1)
                    
                    # For timeline visualization
                    start_date = timestamp
                    end_date = timestamp + pd.Timedelta(days=np.random.randint(1, 15))  # Effect duration
                    
                    mock_data.append({
                        "timestamp": timestamp,
                        "strategy": strategy,
                        "event_type": event_type,
                        "description": desc,
                        "impact": f"{impact} ({impact_value:.1f}%)",
                        "start_date": start_date,
                        "end_date": end_date
                    })
                
                # Sort by timestamp
                mock_data = sorted(mock_data, key=lambda x: x["timestamp"], reverse=True)
                return pd.DataFrame(mock_data)
            else:
                # Direct MongoDB access
                db = st.session_state["db"]
                adaptation = db.strategy_states.find_one({"strategy_id": "strategy_adaptation"})
                
                if adaptation and "data" in adaptation and "events" in adaptation["data"]:
                    return pd.DataFrame(adaptation["data"]["events"])
                    
                # If no data in database, generate mock data for demonstration
                now = pd.Timestamp.now()
                strategies = ["TrendFollowing", "MeanReversion", "Breakout", "Momentum", "Scalping"]
                event_types = ["Parameter Adjustment", "Regime Change Response", "Risk Adjustment", "New Feature", "Filter Added"]
                impacts = ["Positive", "Neutral", "Negative"]
                
                mock_data = []
                for i in range(15):
                    days_ago = np.random.randint(1, 60)
                    timestamp = now - pd.Timedelta(days=days_ago)
                    strategy = np.random.choice(strategies)
                    event_type = np.random.choice(event_types)
                    
                    # Create realistic descriptions
                    if event_type == "Parameter Adjustment":
                        desc = f"Adjusted {strategy} lookback period from {np.random.randint(10, 30)} to {np.random.randint(10, 30)} days"
                    elif event_type == "Regime Change Response":
                        desc = f"Switched {strategy} to {np.random.choice(['conservative', 'aggressive', 'neutral'])} mode due to regime change"
                    elif event_type == "Risk Adjustment":
                        desc = f"Modified {strategy} risk per trade from {np.random.uniform(0.5, 2.0):.1f}% to {np.random.uniform(0.5, 2.0):.1f}%"
                    elif event_type == "New Feature":
                        desc = f"Added {np.random.choice(['volatility filter', 'volume confirmation', 'news filter', 'correlation check'])} to {strategy}"
                    elif event_type == "Filter Added":
                        desc = f"Implemented {np.random.choice(['ADX filter', 'RSI filter', 'Bollinger filter', 'MA crossover filter'])} to {strategy}"
                    
                    impact = np.random.choice(impacts, p=[0.6, 0.3, 0.1])  # Bias toward positive outcomes
                    impact_value = np.random.uniform(-5, 15) if impact == "Positive" else \
                                  np.random.uniform(-2, 2) if impact == "Neutral" else \
                                  np.random.uniform(-12, -1)
                    
                    # For timeline visualization
                    start_date = timestamp
                    end_date = timestamp + pd.Timedelta(days=np.random.randint(1, 15))  # Effect duration
                    
                    mock_data.append({
                        "timestamp": timestamp,
                        "strategy": strategy,
                        "event_type": event_type,
                        "description": desc,
                        "impact": f"{impact} ({impact_value:.1f}%)",
                        "start_date": start_date,
                        "end_date": end_date
                    })
                
                # Sort by timestamp
                mock_data = sorted(mock_data, key=lambda x: x["timestamp"], reverse=True)
                return pd.DataFrame(mock_data)
    except Exception as e:
        logger.error(f"Error getting strategy adaptation data: {str(e)}")
        return None

if __name__ == "__main__":
    main()
