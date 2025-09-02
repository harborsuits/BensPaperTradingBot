"""
Standalone launcher for the trading component.
This provides a clean, separate interface to launch and monitor your trading bot.
"""

import streamlit as st
import os
import logging
from trading_bot.ui.trading_launcher import TradingLauncher

# Configure the page
st.set_page_config(
    page_title="Trading Bot Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading_launcher")

# Title and description
st.title("🚀 Trading Bot Control Center")

st.write("""
### Welcome to your Trading Bot Control Center

This standalone interface gives you direct control over your trading bot, allowing you to:
- Configure trading parameters
- Start and stop automated trading
- Monitor positions and performance in real-time 
- View trading logs and activity
""")

# Try to import and initialize the trading launcher
try:
    # Initialize trading launcher
    trading_launcher = TradingLauncher()
    
    # Render the launcher UI
    trading_launcher.render()

except ImportError as e:
    st.error(f"Error: Could not import required components: {e}")
    
    st.info("""
    The trading launcher requires additional modules to be installed.
    Please make sure all required components are installed or contact support for assistance.
    """)
    
except Exception as e:
    st.error(f"Error initializing trading launcher: {str(e)}")
    st.info("Please check the logs for details or contact support for assistance.")

# Display helpful information in the sidebar
with st.sidebar:
    st.header("Trading System Info")
    
    st.info("""
    **Trading Bot Components**
    - Data Provider: Yahoo Finance
    - Strategies: Momentum, Mean Reversion, Trend Following, Volatility Breakout
    - Risk Management: Position sizing, stop-loss, take-profit
    - Portfolio Tracking: Real-time position monitoring
    """)
    
    st.markdown("---")
    
    st.markdown("""
    **Quick Guide**
    1. Configure your trading parameters
    2. Click 'Start Trading' to begin
    3. Monitor your positions and performance
    4. Use 'Stop Trading' to halt operations
    """)
