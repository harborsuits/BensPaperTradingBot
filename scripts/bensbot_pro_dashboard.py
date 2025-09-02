"""
BensBot Professional Trading Dashboard
A modern, professional-grade dashboard for trading monitoring and control
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import pymongo
import os
import sys
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="BensBot Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard component imports
from dashboard_components import overview
from dashboard_components import strategy_library
from dashboard_components import evotrader
from dashboard_components import market_context
from dashboard_components.orders_positions_ui import render as render_orders_positions
from dashboard_components.signals_approvals_ui import render as render_signals_approvals

# Apply professional trading platform styling 
def apply_professional_styling():
    """Apply custom styling to make the dashboard look like a professional trading platform"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --bg-color: #14191f;
        --sidebar-bg: #0d1117;
        --card-bg: #1c2128;
        --accent-color: #2d81ff;
        --text-color: #e6e6e6;
        --border-color: #30363d;
        --success-color: #238636;
        --warning-color: #9e6a03;
        --danger-color: #f85149;
        --chart-grid: rgba(110, 118, 129, 0.1);
    }
    
    /* Main elements */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Sidebar styling */
    header {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .block-container {
        background-color: var(--sidebar-bg);
    }
    
    /* Dashboard cards */
    div.dashboard-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    div.dashboard-card h3 {
        color: var(--text-color);
        font-size: 1.2rem;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Navigation styling */
    .nav-button {
        width: 100%;
        text-align: left;
        padding: 10px;
        border: none;
        background-color: transparent;
        color: var(--text-color);
        font-size: 1rem;
        cursor: pointer;
        margin-bottom: 5px;
        border-radius: 4px;
    }
    
    .nav-button:hover {
        background-color: rgba(45, 129, 255, 0.1);
    }
    
    .nav-button.active {
        background-color: rgba(45, 129, 255, 0.2);
        color: var(--accent-color);
        font-weight: bold;
    }
    
    /* Metric tiles */
    .metric-tile {
        background-color: var(--card-bg);
        padding: 10px;
        border-radius: 4px;
        border: 1px solid var(--border-color);
        text-align: center;
    }
    
    .metric-tile p.label {
        color: #8b949e;
        font-size: 0.8rem;
        margin-bottom: 0;
    }
    
    .metric-tile p.value {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .metric-tile p.change {
        font-size: 0.9rem;
        margin-top: 0;
    }
    
    .metric-tile p.change.positive {
        color: var(--success-color);
    }
    
    .metric-tile p.change.negative {
        color: var(--danger-color);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 4px;
        border: 1px solid var(--border-color);
        background-color: var(--card-bg);
        color: var(--text-color);
    }
    
    .stButton button:hover {
        border: 1px solid var(--accent-color);
        background-color: rgba(45, 129, 255, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--card-bg);
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 4px 4px 0 0;
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-bottom: none;
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(45, 129, 255, 0.1);
        border-top: 2px solid var(--accent-color);
    }
    
    /* Dataframes/tables */
    .dataframe {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
    }
    
    .dataframe th {
        background-color: var(--sidebar-bg);
        color: var(--text-color);
    }
    
    .dataframe td {
        color: var(--text-color);
        background-color: var(--card-bg);
    }
    
    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stMultiselect {
        background-color: var(--card-bg);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    
    /* Additional classes for dashboard layout */
    .main-header {
        background-color: var(--sidebar-bg);
        padding: 10px 20px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 16px;
        border-radius: 4px;
    }
    
    .account-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 8px;
    }
    
    .badge-live {
        background-color: var(--danger-color);
        color: white;
    }
    
    .badge-paper {
        background-color: var(--warning-color);
        color: white;
    }
    
    .badge-backtest {
        background-color: var(--success-color);
        color: white;
    }
    
    .badge-all {
        background-color: var(--accent-color);
        color: white;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: var(--text-color);
        margin-bottom: 10px;
    }
    
    /* Plot styling */
    .js-plotly-plot {
        background-color: var(--card-bg) !important;
    }
    
    .js-plotly-plot .plot-container .svg-container {
        background-color: var(--card-bg) !important;
    }
    
    /* Override Streamlit elements */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    h1, h2, h3, h4, h5, h6, a {
        color: var(--text-color) !important;
    }
    
    div[data-testid="stMarkdownContainer"] > p {
        color: var(--text-color) !important;
    }
    
    li {
        color: var(--text-color) !important;
    }
    
    div[role="radiogroup"] label, div[data-testid="stForm"] label, div[data-testid="stWidgetLabel"] {
        color: var(--text-color) !important;
    }
    
    button[data-testid="baseButton-secondary"], div[data-testid="stFormSubmitButton"] > button {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    button[data-testid="baseButton-secondary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        border: 1px solid var(--accent-color) !important;
        background-color: rgba(45, 129, 255, 0.1) !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# MongoDB connection setup
def connect_to_mongo():
    """Connect to MongoDB and return database client"""
    try:
        # Get MongoDB URI from environment variable or use default
        mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Check connection
        client.server_info()
        db = client.bensbot  # Use 'bensbot' database
        
        return db, None
    except Exception as e:
        return None, str(e)

# Custom header with account status
def render_header(account_type):
    """Render professional header with account selection and status"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dashboard title with selected account badge
        badge_class = {
            "All Accounts": "badge-all",
            "Live Trading": "badge-live",
            "Paper Trading": "badge-paper",
            "Backtest": "badge-backtest"
        }.get(account_type, "badge-all")
        
        st.markdown(f"""
        <div class="main-header">
            <h1 style="margin: 0; display: inline-block; vertical-align: middle;">
                BensBot Pro
                <span class="account-badge {badge_class}">{account_type}</span>
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Current time and status indicators
        now = datetime.datetime.now()
        
        # Mongo connection status
        db, mongo_error = connect_to_mongo()
        mongo_status = "🟢 Connected" if db is not None else f"🔴 Disconnected"
        
        # Market status - mock data
        current_hour = now.hour
        is_market_open = 9 <= current_hour < 16  # Simple 9am-4pm market hours
        market_status = "🟢 Open" if is_market_open else "🔴 Closed"
        
        # Trading status
        trading_active = True  # Mock status, would be from the real system
        trading_status = "🟢 Active" if trading_active else "🔴 Inactive"
        
        st.markdown(f"""
        <div style="text-align: right; padding: 10px 0;">
            <p style="margin: 0; color: #8b949e; font-size: 0.9rem;">
                {now.strftime('%Y-%m-%d %H:%M:%S')}
            </p>
            <p style="margin: 5px 0; font-size: 0.9rem;">
                MongoDB: {mongo_status}
            </p>
            <p style="margin: 5px 0; font-size: 0.9rem;">
                Market: {market_status} | Trading: {trading_status}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    return db

# Sidebar with advanced navigation
def render_sidebar():
    """Render sidebar with professional navigation"""
    # Account type selection
    st.sidebar.markdown("## Account Selection")
    account_type = st.sidebar.radio(
        "",
        ["All Accounts", "Live Trading", "Paper Trading", "Backtest"],
        horizontal=False,
        label_visibility="collapsed"
    )
    
    # Navigation sections with emojis
    st.sidebar.markdown("## Navigation")
    
    sections = {
        "Overview": "🏠",
        "Strategy Library": "⚙️",
        "EvoTrader": "🧬",
        "Market Context": "📊",
        "Orders & Positions": "💼",
        "Signals & Approvals": "🔔",
        "Logs": "📜",
        "Settings": "⚙️"
    }
    
    # Create nav buttons
    selected_section = None
    
    for section, emoji in sections.items():
        # Check if this section is currently active
        is_active = (st.session_state.get('selected_section') == section)
        active_class = "active" if is_active else ""
        
        # Create a custom HTML button with proper styling
        if st.sidebar.markdown(f"""
        <button class="nav-button {active_class}" id="nav_{section}">
            {emoji} {section}
        </button>
        """, unsafe_allow_html=True):
            selected_section = section
    
    # JavaScript to handle button clicks
    st.sidebar.markdown("""
    <script>
    document.querySelectorAll('.nav-button').forEach(button => {
        button.addEventListener('click', function() {
            // Get the section name from the button ID
            const section = this.id.replace('nav_', '');
            
            // Update the URL query parameter
            const url = new URL(window.location.href);
            url.searchParams.set('section', section);
            window.history.pushState({}, '', url);
            
            // Reload the page to apply the change
            window.location.reload();
        });
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Get section from URL parameter or session state or default to Overview
    query_params = st.experimental_get_query_params()
    url_section = query_params.get("section", [None])[0]
    
    if url_section in sections:
        selected_section = url_section
    elif selected_section is None:
        selected_section = st.session_state.get('selected_section', "Overview")
    
    # Store in session state
    st.session_state['selected_section'] = selected_section
    
    # Auto refresh settings (collapsible)
    with st.sidebar.expander("Auto Refresh"):
        auto_refresh = st.checkbox("Enable Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Interval (seconds)",
                min_value=5,
                max_value=60,
                value=30
            )
            
            # Add auto-refresh using JavaScript
            st.markdown(
                f"""
                <script>
                    var refreshInterval = {refresh_interval * 1000};
                    setTimeout(function() {{
                        window.location.reload();
                    }}, refreshInterval);
                </script>
                """,
                unsafe_allow_html=True
            )
    
    # Key metrics overview
    st.sidebar.markdown("## Key Metrics")
    
    # Mock data for sidebar metrics
    account_balance = 124680.45
    daily_pnl = 1245.78
    daily_pnl_pct = (daily_pnl / account_balance) * 100
    positions_count = 8
    win_rate = 68.5
    
    # Display metrics in a grid
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-tile">
            <p class="label">Account Balance</p>
            <p class="value">${account_balance:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-tile">
            <p class="label">Open Positions</p>
            <p class="value">{positions_count}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pnl_class = "positive" if daily_pnl >= 0 else "negative"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        
        st.markdown(f"""
        <div class="metric-tile">
            <p class="label">Daily P&L</p>
            <p class="value">${abs(daily_pnl):,.2f}</p>
            <p class="change {pnl_class}">{pnl_sign}{daily_pnl_pct:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-tile">
            <p class="label">Win Rate</p>
            <p class="value">{win_rate}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trading controls
    st.sidebar.markdown("## Trading Controls")
    
    # Trading mode buttons
    trading_col1, trading_col2 = st.sidebar.columns(2)
    
    with trading_col1:
        if st.button("🟢 Start Trading", use_container_width=True):
            st.sidebar.success("Trading activated")
    
    with trading_col2:
        if st.button("🔴 Stop Trading", use_container_width=True):
            st.sidebar.error("Trading stopped")
    
    # Emergency stop button
    if st.sidebar.button("⚠️ EMERGENCY STOP", use_container_width=True):
        st.sidebar.warning("Emergency stop activated - All positions would be closed")
    
    # Dashboard version info
    st.sidebar.markdown("---")
    st.sidebar.caption("BensBot Pro Dashboard v1.0")
    st.sidebar.caption(f"System time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return account_type, selected_section

# Dashboard component rendering
def render_dashboard_section(db, account_type, section):
    """Render the selected dashboard section with pro styling"""
    
    # Custom section header
    st.markdown(f"""
    <div class="section-title">{section}</div>
    """, unsafe_allow_html=True)
    
    # Display the selected section
    if section == "Overview":
        overview.render(db, account_type)
    
    elif section == "Strategy Library":
        strategy_library.render(db, account_type)
    
    elif section == "EvoTrader":
        evotrader.render(db)
    
    elif section == "Market Context":
        market_context.render(db)
    
    elif section == "Orders & Positions":
        render_orders_positions(db, account_type)
    
    elif section == "Signals & Approvals":
        render_signals_approvals(db, account_type)
    
    elif section == "Logs":
        render_logs(db, account_type)
    
    elif section == "Settings":
        render_settings(db)
    
    else:
        st.warning(f"Section '{section}' not implemented yet")

def render_logs(db, account_type):
    """Render the logs section with pro styling"""
    st.markdown("""
    <div class="dashboard-card">
        <h3>System Logs</h3>
        <div style="color: #8b949e;">
            This section will include combined log viewer with filtering, error highlighting,
            and system health monitoring. Coming in the next update.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_settings(db):
    """Render the settings section with pro styling"""
    st.markdown("""
    <div class="dashboard-card">
        <h3>System Settings</h3>
        <div style="color: #8b949e;">
            This section will include dashboard preferences, alert configuration,
            API key management, and system backup options. Coming in the next update.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main app function
def main():
    """Main function to run the professional-grade dashboard"""
    # Apply professional styling
    apply_professional_styling()
    
    # Initialize session state for navigation
    if 'selected_section' not in st.session_state:
        st.session_state['selected_section'] = "Overview"
    
    # Create sidebar and get selected account type and section
    account_type, selected_section = render_sidebar()
    
    # Render header with account selection
    db = render_header(account_type)
    
    # If MongoDB is not connected, show error message
    if db is None:
        st.error("Cannot connect to MongoDB database. The dashboard will run with mock data.")
    
    # Render the dashboard section
    render_dashboard_section(db, account_type, selected_section)

# Run the app
if __name__ == "__main__":
    main()
