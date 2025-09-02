"""
Simplified Trading Dashboard with Modular Strategy Tab

This is a streamlined version of the trading dashboard that focuses on the
modular strategy functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import uuid
import logging
from datetime import datetime, timedelta
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular strategy simplified UI
try:
    from trading_bot.ui.modular_strategy_simplified import ModularStrategySimplifiedUI
    modular_strategy_available = True
except ImportError as e:
    modular_strategy_available = False
    st.warning(f"Modular strategy UI not available: {e}")

# Page title and header
st.title("Trading Bot Dashboard")
st.markdown('<div class="title-area"><h1>Institutional-Grade Trading Platform</h1></div>', unsafe_allow_html=True)

# Add a simple sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Dashboard", "Modular Strategy System"])
    
    st.header("Settings")
    risk_level = st.select_slider("Risk Level", options=["Low", "Medium", "High"], value="Medium")
    
    # Mock portfolio data
    portfolio_value = 100000 + np.random.normal(0, 5000)
    cash = 25000 + np.random.normal(0, 2000)
    
    st.metric("Portfolio Value", f"${portfolio_value:.2f}")
    st.metric("Available Cash", f"${cash:.2f}")

# Main content area
if page == "Dashboard":
    st.header("Dashboard Overview")
    
    # Mock metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Win Rate", "67.8%", "2.1%")
    with metrics_col2:
        st.metric("Profit Factor", "1.87", "0.12")
    with metrics_col3:
        st.metric("Max Drawdown", "-4.2%", "0.8%", delta_color="inverse")
    with metrics_col4:
        st.metric("Sharpe Ratio", "1.93", "0.05")
    
    # Mock charts
    st.subheader("Portfolio Performance")
    
    # Generate mock data for a performance chart
    days = 90
    dates = pd.date_range(end=datetime.now(), periods=days)
    
    # Create a portfolio value series with some randomness but overall uptrend
    np.random.seed(42)  # For reproducibility
    portfolio_values = 100000 + np.cumsum(np.random.normal(100, 500, days))
    benchmark_values = 100000 + np.cumsum(np.random.normal(50, 400, days))
    
    # Create DataFrame
    performance_df = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_values,
        'Benchmark (S&P 500)': benchmark_values
    })
    
    # Convert to percentage change for better visualization
    performance_df['Portfolio'] = (performance_df['Portfolio'] / performance_df['Portfolio'].iloc[0] - 1) * 100
    performance_df['Benchmark (S&P 500)'] = (performance_df['Benchmark (S&P 500)'] / performance_df['Benchmark (S&P 500)'].iloc[0] - 1) * 100
    
    # Display the chart
    st.line_chart(performance_df.set_index('Date'))
    
    # Mock positions table
    st.subheader("Current Positions")
    
    positions_data = {
        "Symbol": ["AAPL", "MSFT", "TSLA", "AMZN", "SPY"],
        "Quantity": [25, 15, 10, 5, 20],
        "Entry Price": ["$145.32", "$280.10", "$225.65", "$135.42", "$415.20"],
        "Current Price": ["$162.45", "$310.25", "$245.30", "$142.18", "$440.75"],
        "Profit/Loss": ["11.8%", "10.8%", "8.7%", "5.0%", "6.2%"],
        "Strategy": ["Momentum", "Growth", "Momentum", "Value", "Index"]
    }
    
    positions_df = pd.DataFrame(positions_data)
    st.dataframe(positions_df, use_container_width=True)
    
    # Mock recent trades
    st.subheader("Recent Trades")
    
    trades_data = {
        "Symbol": ["GOOGL", "QQQ", "NVDA", "FB", "DIS"],
        "Type": ["BUY", "SELL", "BUY", "SELL", "BUY"],
        "Quantity": [10, 15, 20, 25, 30],
        "Price": ["$2,815.24", "$368.45", "$275.60", "$325.18", "$142.75"],
        "Date": ["2023-04-20", "2023-04-18", "2023-04-15", "2023-04-12", "2023-04-10"],
        "Strategy": ["Growth", "Rebalance", "Momentum", "Take Profit", "Value"]
    }
    
    trades_df = pd.DataFrame(trades_data)
    st.dataframe(trades_df, use_container_width=True)

elif page == "Modular Strategy System":
    # Modular Strategy System
    st.header("Modular Strategy Builder")
    
    if modular_strategy_available:
        try:
            # Initialize and render the simplified UI
            simplified_ui = ModularStrategySimplifiedUI()
            simplified_ui.render()
        except Exception as e:
            st.error(f"Error initializing modular strategy system: {e}")
            st.exception(e)
    else:
        st.warning("Modular strategy system is not available. Please check that all required components are installed.")
        
        # Show placeholder UI with explanation of modular strategy system
        st.markdown("""
        ## Modular Strategy System
        
        The modular strategy system allows you to create trading strategies by combining:
        
        * **Signal Generators** - Produce buy/sell signals based on indicators or patterns
        * **Filters** - Validate signals based on market conditions
        * **Position Sizers** - Determine optimal position size for each trade
        * **Exit Managers** - Control when to exit positions
        
        The modular strategy system integrates with the event-driven architecture to provide:
        * Component-level performance analytics
        * Parameter optimization for strategy components
        * Component marketplace for sharing and importing components
        * Comprehensive testing and validation tools
        
        Install the required components to enable this feature.
        """)

# Add footer
st.markdown("---")
st.markdown("Trading Bot Dashboard | © 2025 | Institutional-Grade Trading Platform")
