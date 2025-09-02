"""
BensBot Professional Trading Dashboard
A comprehensive trading platform interface with strategy management,
genetic algorithm visualization, and performance monitoring.
"""
import os
import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
import json
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="BensBot Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    /* Base styling */
    .main { padding-top: 2rem; }
    .stApp { background-color: #f8fafc; }
    
    /* Navigation styling */
    .nav-section {
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
        cursor: pointer;
        display: flex;
        align-items: center;
    }
    .nav-section:hover { background-color: rgba(0,0,0,0.05); }
    .nav-section.active { 
        background-color: #1e40af; 
        color: white;
        font-weight: 600;
    }
    .nav-icon { 
        font-size: 1.2rem;
        margin-right: 0.5rem;
        width: 1.5rem;
        text-align: center;
    }
    
    /* Card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetricDelta"] { font-size: 0.85rem !important; font-weight: 600 !important; }
    
    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 8px;
        overflow: hidden;
    }
    .styled-table thead tr {
        background-color: #1e40af;
        color: #ffffff;
        text-align: left;
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #1e40af;
    }
    .styled-table tbody tr.active-row {
        font-weight: bold;
        color: #1e40af;
    }
    
    /* Button styling */
    .button-approve {
        background-color: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
    }
    .button-reject {
        background-color: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
    }
    .button-neutral {
        background-color: #6b7280;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }
    .status-offline {
        color: #ef4444;
        font-weight: 600;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e40af;
        font-weight: 700;
        border-top: 3px solid #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MongoDB connection
@st.cache_resource(ttl=300)
def connect_to_mongodb():
    try:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/bensbot_trading")
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test connection
        db = client.get_database()
        return db, True
    except Exception as e:
        st.sidebar.error(f"MongoDB connection error: {e}")
        return None, False

# Get real-time market data
@st.cache_data(ttl=60)
def get_market_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if len(data) > 0:
            price = data.iloc[-1]['Close']
            change = ((price / data.iloc[0]['Open']) - 1) * 100
            return price, change
        return None, None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None, None

# Navigation items with emoji icons
NAV_ITEMS = {
    "Overview": {"icon": "🏠", "title": "Overview"},
    "Strategy_Library": {"icon": "⚙️", "title": "Strategy Library"},
    "EvoTrader": {"icon": "🧬", "title": "EvoTrader (Genetic Module)"},
    "Market_Context": {"icon": "📊", "title": "Market Context"},
    "Orders_Positions": {"icon": "💼", "title": "Orders & Positions"},
    "Signals_Approvals": {"icon": "🔔", "title": "Signals & Approvals"},
    "Logs": {"icon": "📜", "title": "Logs"},
    "Settings": {"icon": "⚙️", "title": "Settings"}
}

# Initialize session state for navigation
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = "Overview"

# Create navigation sidebar
def render_sidebar():
    st.sidebar.title("BensBot Trading")
    
    # MongoDB connection status
    db, connected = connect_to_mongodb()
    if connected:
        st.sidebar.success("✅ Connected to MongoDB")
    else:
        st.sidebar.error("❌ MongoDB Not Connected")
        st.sidebar.info("Using demo data")
    
    # Account type selector
    account_type = st.sidebar.selectbox(
        "Account Type",
        ["All Accounts", "Paper Trading", "Live Trading", "Backtest"],
        index=0
    )
    
    # Navigation menu
    st.sidebar.markdown("## Navigation")
    
    for key, item in NAV_ITEMS.items():
        is_active = st.session_state.nav_selection == key
        active_class = "active" if is_active else ""
        
        # Create clickable navigation item
        nav_html = f"""
        <div class="nav-section {active_class}" onclick="handleNavClick('{key}')">
            <span class="nav-icon">{item["icon"]}</span>
            <span>{item["title"]}</span>
        </div>
        """
        st.sidebar.markdown(nav_html, unsafe_allow_html=True)
    
    # Create a hidden button for each nav item to handle clicks
    for key in NAV_ITEMS:
        if st.sidebar.button(f"btn_{key}", key=f"btn_{key}", label_visibility="collapsed"):
            st.session_state.nav_selection = key
            st.rerun()
    
    # JavaScript for handling navigation clicks
    st.sidebar.markdown("""
    <script>
    function handleNavClick(section) {
        // Find the hidden button and click it
        const buttons = window.parent.document.querySelectorAll('button');
        for (const button of buttons) {
            if (button.innerText === `btn_${section}`) {
                button.click();
            }
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.sidebar.caption("Dashboard will refresh automatically")
        st.markdown(
            """
            <script>
                setTimeout(function() {
                    window.location.reload();
                }, 30000);
            </script>
            """,
            unsafe_allow_html=True
        )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Return the database connection and selected account type
    return db, account_type

# Placeholder functions for each section
# Each section will be implemented fully in subsequent steps

def render_overview(db, account_type):
    st.title("🏠 Overview")
    st.markdown("### Performance Dashboard")
    
    # Placeholder for equity curve
    st.info("Equity curve will be displayed here")
    
    # Placeholder for regime & risk posture
    st.markdown("### Market Regime & Risk Posture")
    st.info("Market regime indicators will be displayed here")
    
    # Placeholder for quick stats
    st.markdown("### Performance Comparison")
    st.info("Quick-stats comparing live, paper, and backtest performance will be displayed here")

def render_strategy_library(db, account_type):
    st.title("⚙️ Strategy Library")
    
    # Create tabs for different account types
    tabs = st.tabs(["Backtest", "Paper Test", "Live"])
    
    with tabs[0]:
        st.markdown("### Backtest Strategies")
        st.info("Historical results and charts for backtest strategies will be displayed here")
    
    with tabs[1]:
        st.markdown("### Paper Test Strategies")
        st.info("Simulated paper trading results and current paper positions will be displayed here")
    
    with tabs[2]:
        st.markdown("### Live Strategies")
        st.markdown("Real money P&L and open positions will be displayed here")

def render_evotrader(db, account_type):
    st.title("🧬 EvoTrader (Genetic Module)")
    
    # Placeholder for generation slider
    st.markdown("### Generation History")
    st.info("Generation slider and visualization will be displayed here")
    
    # Placeholder for population table
    st.markdown("### Population Metrics")
    st.info("Strategy population table with fitness scores will be displayed here")
    
    # Placeholder for controls
    st.markdown("### Evolution Controls")
    st.info("Strategy promotion/discard/re-evolve controls will be displayed here")

def render_market_context(db, account_type):
    st.title("📊 Market Context")
    
    # Placeholder for market data
    st.info("Market sentiment, news, VIX data, and heatmaps will be displayed here")

def render_orders_positions(db, account_type):
    st.title("💼 Orders & Positions")
    
    # Create tabs for orders and positions
    tabs = st.tabs(["Open Positions", "Orders"])
    
    with tabs[0]:
        st.markdown("### Open Positions")
        st.info("Current open positions with P&L will be displayed here")
    
    with tabs[1]:
        st.markdown("### Orders")
        st.info("Order history and status will be displayed here")

def render_signals_approvals(db, account_type):
    st.title("🔔 Signals & Approvals")
    
    # Placeholder for signals
    st.markdown("### Upcoming Signals")
    st.info("Upcoming trade signals will be displayed here")
    
    # Placeholder for approvals
    st.markdown("### Pending Approvals")
    st.info("Strategies awaiting approval will be displayed here")

def render_logs(db, account_type):
    st.title("📜 Logs")
    
    # Placeholder for log viewer
    st.info("Combined logs with filtering will be displayed here")

def render_settings(db, account_type):
    st.title("⚙️ Settings")
    
    # Placeholder for settings
    st.info("System settings and configuration will be displayed here")

# Main application
def main():
    # Render sidebar and get database connection and account type
    db, account_type = render_sidebar()
    
    # Render the selected section
    if st.session_state.nav_selection == "Overview":
        render_overview(db, account_type)
    elif st.session_state.nav_selection == "Strategy_Library":
        render_strategy_library(db, account_type)
    elif st.session_state.nav_selection == "EvoTrader":
        render_evotrader(db, account_type)
    elif st.session_state.nav_selection == "Market_Context":
        render_market_context(db, account_type)
    elif st.session_state.nav_selection == "Orders_Positions":
        render_orders_positions(db, account_type)
    elif st.session_state.nav_selection == "Signals_Approvals":
        render_signals_approvals(db, account_type)
    elif st.session_state.nav_selection == "Logs":
        render_logs(db, account_type)
    elif st.session_state.nav_selection == "Settings":
        render_settings(db, account_type)

if __name__ == "__main__":
    main()
