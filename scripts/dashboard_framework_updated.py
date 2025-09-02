"""
BensBot Professional Trading Dashboard
A multi-section dashboard for monitoring and controlling automated trading operations
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
from streamlit.web.server.websocket_headers import _get_websocket_headers

# Set page configuration
st.set_page_config(
    page_title="BensBot Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard component imports
try:
    from dashboard_components import overview
    from dashboard_components import strategy_library
    from dashboard_components import evotrader
    from dashboard_components import market_context
    from dashboard_components.orders_positions import get_positions, get_open_orders, get_execution_history
    from dashboard_components.orders_positions_ui import render as render_orders_positions
    from dashboard_components.signals_approvals import get_trading_signals, get_pending_approvals
    from dashboard_components.signals_approvals_ui import render as render_signals_approvals
except ImportError as e:
    st.error(f"Error importing dashboard components: {e}")
    st.info("Make sure all components are properly installed and in the correct directory.")

# Load custom CSS theme
def load_custom_css():
    """Load custom CSS styling for the dashboard"""
    try:
        with open(os.path.join("ui_assets", "custom_theme.css"), "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom CSS: {e}")
        # Fallback to in-line basic styling
        st.markdown("""
        <style>
        /* Basic theming for trading dashboard */
        .stApp {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        h1, h2, h3 {
            color: #343a40;
        }
        .metric-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            padding: 15px;
            margin-bottom: 15px;
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
        
        st.sidebar.success("✅ Connected to MongoDB")
        return db
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB Connection Error: {e}")
        return None

# Sidebar navigation
def create_sidebar():
    """Create sidebar navigation with section selection"""
    st.sidebar.title("BensBot Pro")
    st.sidebar.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    
    # Account type selection
    account_type = st.sidebar.radio(
        "Account Type",
        ["All Accounts", "Paper Trading", "Live Trading", "Backtest"]
    )
    
    # Horizontal divider
    st.sidebar.markdown("---")
    
    # Navigation sections with emojis
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
    
    # Create list-like navigation
    selected_section = None
    
    for section, emoji in sections.items():
        if st.sidebar.button(f"{emoji} {section}", key=f"nav_{section}"):
            selected_section = section
    
    # If nothing is selected, default to Overview
    if selected_section is None:
        selected_section = "Overview"
    
    # Session state to store selected section
    if "selected_section" not in st.session_state:
        st.session_state.selected_section = selected_section
    
    # Update session state if selection changed
    if selected_section != st.session_state.selected_section:
        st.session_state.selected_section = selected_section
    
    # Horizontal divider
    st.sidebar.markdown("---")
    
    # Auto refresh toggle and manual refresh button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Auto refresh interval if enabled
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (s)",
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
    
    # MongoDB connection status
    st.sidebar.markdown("---")
    
    # Dashboard information
    st.sidebar.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption("BensBot Pro Dashboard v1.0")
    
    return account_type, st.session_state.selected_section

# Render dashboard sections
def render_dashboard(db, account_type, section):
    """Render the selected dashboard section"""
    
    # Page header
    st.title(f"{section}")
    
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
    """Render the logs section"""
    st.info("Logs section will be implemented in the next phase.")
    
    # Placeholder function - implemented in future
    st.markdown("""
    ### Coming Soon: Logs Dashboard
    
    This section will include:
    - Combined log viewer with source filtering
    - Error highlighting and resolution suggestions
    - Log search and export functionality
    - System health monitoring
    """)

def render_settings(db):
    """Render the settings section"""
    st.info("Settings section will be implemented in the next phase.")
    
    # Placeholder function - implemented in future
    st.markdown("""
    ### Coming Soon: System Settings
    
    This section will include:
    - Dashboard preferences
    - Alert configuration
    - API key management
    - System backup and restore options
    """)

# Main app function
def main():
    """Main function to run the dashboard"""
    # Load custom CSS
    load_custom_css()
    
    # Connect to MongoDB
    db = connect_to_mongo()
    
    # Create sidebar and get selected account type and section
    account_type, selected_section = create_sidebar()
    
    # Render the dashboard
    render_dashboard(db, account_type, selected_section)

# Run the app
if __name__ == "__main__":
    main()
