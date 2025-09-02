"""
Simple Trading Control Panel - A streamlined interface to control trading operations
"""

import streamlit as st
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta
import threading
import logging
import random
from typing import Dict, List, Any

# Configure the page
st.set_page_config(
    page_title="Trading Control Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading_control")

# Initialize session state
if "trading_active" not in st.session_state:
    st.session_state.trading_active = False
if "trading_thread" not in st.session_state:
    st.session_state.trading_thread = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        "cash": 10000.0,
        "total_value": 10000.0,
        "positions": {}
    }
if "settings" not in st.session_state:
    st.session_state.settings = {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "strategy": "Momentum",
        "risk_level": 2,
        "position_size_pct": 10,
        "use_stoploss": True,
        "stoploss_pct": 5,
        "use_takeprofit": True,
        "takeprofit_pct": 15
    }

# Title and welcome message
st.title("🚀 Trading Control Center")
st.write("Configure, start, and monitor your trading operations from this centralized control panel.")

# Create two main columns
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Trading Configuration")
    
    with st.form("trading_settings"):
        st.subheader("Basic Settings")
        symbols = st.multiselect(
            "Trading Symbols",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ"],
            st.session_state.settings["symbols"]
        )
        
        strategy = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion", "Trend Following", "Volatility Breakout"],
            index=["Momentum", "Mean Reversion", "Trend Following", "Volatility Breakout"].index(st.session_state.settings["strategy"])
        )
        
        st.subheader("Risk Parameters")
        
        risk_level = st.slider(
            "Risk Level", 
            1, 5, 
            st.session_state.settings["risk_level"],
            help="1: Very Conservative, 5: Aggressive"
        )
        
        position_size_pct = st.slider(
            "Max Position Size (%)", 
            1, 100, 
            st.session_state.settings["position_size_pct"],
            help="Maximum percentage of capital for a single position"
        )
        
        with st.expander("Advanced Settings"):
            use_stoploss = st.checkbox(
                "Use Stop Loss", 
                value=st.session_state.settings["use_stoploss"]
            )
            
            stoploss_pct = st.slider(
                "Stop Loss (%)", 
                1, 20, 
                st.session_state.settings["stoploss_pct"]
            )
            
            use_takeprofit = st.checkbox(
                "Use Take Profit", 
                value=st.session_state.settings["use_takeprofit"]
            )
            
            takeprofit_pct = st.slider(
                "Take Profit (%)", 
                1, 50, 
                st.session_state.settings["takeprofit_pct"]
            )
        
        submitted = st.form_submit_button("Apply Settings")
        
        if submitted:
            st.session_state.settings = {
                "symbols": symbols,
                "strategy": strategy,
                "risk_level": risk_level,
                "position_size_pct": position_size_pct,
                "use_stoploss": use_stoploss,
                "stoploss_pct": stoploss_pct,
                "use_takeprofit": use_takeprofit,
                "takeprofit_pct": takeprofit_pct
            }
            st.success("Settings applied successfully!")
    
    # Trading controls
    st.header("Trading Controls")
    
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if not st.session_state.trading_active:
            if st.button("▶️ Start Trading", use_container_width=True):
                st.session_state.trading_active = True
                
                # Log the start
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.logs.append(f"{timestamp} - Trading started with {st.session_state.settings['strategy']} strategy")
                
                # Simulate initial portfolio (for demo)
                simulate_portfolio()
                
                st.experimental_rerun()
    
    with control_col2:
        if st.session_state.trading_active:
            if st.button("⏹️ Stop Trading", use_container_width=True):
                st.session_state.trading_active = False
                
                # Log the stop
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.logs.append(f"{timestamp} - Trading stopped")
                
                st.experimental_rerun()

with col2:
    st.header("Trading Status")
    
    # Status indicator
    status_col, time_col = st.columns(2)
    
    with status_col:
        if st.session_state.trading_active:
            st.markdown("#### Status: 🟢 ACTIVE")
        else:
            st.markdown("#### Status: ⚪ INACTIVE")
    
    with time_col:
        st.markdown(f"#### Time: {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Portfolio overview
    st.subheader("Portfolio Overview")
    
    # Cash and total value
    cash_col, value_col = st.columns(2)
    with cash_col:
        st.metric("Cash Available", f"${st.session_state.portfolio['cash']:,.2f}")
    with value_col:
        st.metric("Total Value", f"${st.session_state.portfolio['total_value']:,.2f}")
    
    # Positions
    st.subheader("Current Positions")
    
    positions = st.session_state.portfolio.get("positions", {})
    if positions:
        positions_df = pd.DataFrame([
            {
                "Symbol": symbol,
                "Quantity": pos.get("quantity", 0),
                "Avg Price": f"${pos.get('avg_price', 0):.2f}",
                "Current Price": f"${pos.get('current_price', 0):.2f}",
                "Current Value": f"${pos.get('current_value', 0):.2f}",
                "P&L": f"${pos.get('unrealized_pnl', 0):.2f}",
                "P&L %": f"{pos.get('unrealized_pnl_pct', 0):.2f}%"
            }
            for symbol, pos in positions.items()
        ])
        
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Trading logs
    st.subheader("Trading Logs")
    
    log_container = st.container()
    with log_container:
        for log in list(st.session_state.logs)[-10:]:
            st.text(log)
    
    # Recent trades
    if st.session_state.trading_active:
        st.subheader("Recent Signals")
        
        # Simulate some signals for demo
        signals_df = pd.DataFrame([
            {
                "Time": (datetime.now() - timedelta(minutes=random.randint(1, 30))).strftime("%H:%M:%S"),
                "Symbol": random.choice(st.session_state.settings["symbols"]),
                "Signal": random.choice(["BUY", "SELL", "HOLD"]),
                "Strength": f"{random.uniform(0.6, 0.95):.2f}",
                "Source": random.choice(["Pattern", "Momentum", "Trend", "Volatility"])
            }
            for _ in range(5)
        ])
        
        st.dataframe(signals_df, use_container_width=True)

# Sidebar with additional info
with st.sidebar:
    st.header("Trading System Info")
    
    st.info("""
    **Active Components**
    - Data Provider: Yahoo Finance
    - Strategy: {}
    - Risk Level: {}
    - Position Sizing: {}%
    """.format(
        st.session_state.settings["strategy"],
        st.session_state.settings["risk_level"],
        st.session_state.settings["position_size_pct"]
    ))
    
    # Show active symbols
    st.subheader("Trading Symbols")
    for symbol in st.session_state.settings["symbols"]:
        st.write(f"• {symbol}")
    
    st.markdown("---")
    
    if st.session_state.trading_active:
        st.success("Trading system is currently active")
        
        # Add a refresh button to simulate activity
        if st.button("Refresh Data"):
            update_portfolio_data()
            st.success("Data refreshed")
    else:
        st.warning("Trading system is currently inactive")
        st.info("Click 'Start Trading' to begin")

# Helper functions 
def simulate_portfolio():
    """Simulate a portfolio for demonstration purposes"""
    symbols = st.session_state.settings["symbols"]
    capital = 10000.0
    risk_level = st.session_state.settings["risk_level"]
    position_size_pct = st.session_state.settings["position_size_pct"]
    
    # Calculate max position size based on risk level and position size setting
    max_position_value = capital * (position_size_pct / 100) * (risk_level / 3)
    
    # Create positions for selected symbols
    positions = {}
    total_position_value = 0
    
    for symbol in symbols:
        # Skip some symbols randomly to make it more realistic
        if random.random() > 0.7:
            continue
            
        # Simulate a position
        base_price = 100 + (hash(symbol) % 200)  # Pseudo-random price based on symbol name
        quantity = int(max_position_value / base_price)
        
        if quantity > 0:
            avg_price = base_price * (1 + random.uniform(-0.02, 0.02))  # Slight variation
            current_price = avg_price * (1 + random.uniform(-0.05, 0.10))  # More variation
            current_value = quantity * current_price
            unrealized_pnl = current_value - (quantity * avg_price)
            unrealized_pnl_pct = (unrealized_pnl / (quantity * avg_price)) * 100
            
            positions[symbol] = {
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct
            }
            
            total_position_value += current_value
            
            # Log the trade
            entry_time = (datetime.now() - timedelta(minutes=random.randint(5, 60))).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.logs.append(f"{entry_time} - BUY {symbol}: {quantity} shares @ ${avg_price:.2f}")
    
    # Update portfolio
    cash_remaining = capital - total_position_value
    
    st.session_state.portfolio = {
        "cash": cash_remaining,
        "total_value": cash_remaining + total_position_value,
        "positions": positions
    }

def update_portfolio_data():
    """Update portfolio data with random price changes to simulate real activity"""
    if not st.session_state.trading_active:
        return
        
    positions = st.session_state.portfolio.get("positions", {})
    total_position_value = 0
    
    # Update prices and values
    for symbol, position in positions.items():
        # Apply a random price change
        price_change_pct = random.uniform(-0.5, 0.8)  # Percentage change
        old_price = position["current_price"]
        new_price = old_price * (1 + price_change_pct/100)
        
        # Update position data
        position["current_price"] = new_price
        position["current_value"] = position["quantity"] * new_price
        position["unrealized_pnl"] = position["current_value"] - (position["quantity"] * position["avg_price"])
        position["unrealized_pnl_pct"] = (position["unrealized_pnl"] / (position["quantity"] * position["avg_price"])) * 100
        
        # Log significant price changes
        if abs(price_change_pct) > 0.2:  # Only log changes > 0.2%
            change_direction = "up" if price_change_pct > 0 else "down"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.logs.append(f"{timestamp} - {symbol} moved {change_direction} {abs(price_change_pct):.2f}% to ${new_price:.2f}")
        
        total_position_value += position["current_value"]
    
    # Occasionally add a trade (buy or sell)
    if random.random() > 0.7:
        simulate_trade()
    
    # Update portfolio totals
    st.session_state.portfolio["total_value"] = st.session_state.portfolio["cash"] + total_position_value

def simulate_trade():
    """Simulate a random trade for demonstration"""
    symbols = st.session_state.settings["symbols"]
    positions = st.session_state.portfolio.get("positions", {})
    
    # Choose a random action
    action = random.choice(["BUY", "SELL"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if action == "BUY":
        # Buy a new position or add to existing
        available_symbols = [s for s in symbols if s not in positions or random.random() > 0.8]
        
        if not available_symbols and positions:
            # Add to existing randomly
            symbol = random.choice(list(positions.keys()))
            position = positions[symbol]
            
            # Add to position
            additional_quantity = max(1, int(position["quantity"] * random.uniform(0.1, 0.3)))
            new_total_quantity = position["quantity"] + additional_quantity
            
            # Calculate new average price (slightly different to simulate market change)
            new_price = position["current_price"] * (1 + random.uniform(-0.01, 0.01))
            old_value = position["quantity"] * position["avg_price"]
            additional_value = additional_quantity * new_price
            new_avg_price = (old_value + additional_value) / new_total_quantity
            
            # Update position
            position["quantity"] = new_total_quantity
            position["avg_price"] = new_avg_price
            position["current_value"] = new_total_quantity * position["current_price"]
            position["unrealized_pnl"] = position["current_value"] - (new_total_quantity * new_avg_price)
            position["unrealized_pnl_pct"] = (position["unrealized_pnl"] / (new_total_quantity * new_avg_price)) * 100
            
            # Deduct from cash
            st.session_state.portfolio["cash"] -= additional_value
            
            # Log the trade
            st.session_state.logs.append(f"{timestamp} - BUY {symbol}: Added {additional_quantity} shares @ ${new_price:.2f}")
            
        elif available_symbols:
            # Buy a new position
            symbol = random.choice(available_symbols)
            
            # Create a new position
            base_price = 100 + (hash(symbol) % 200)
            quantity = max(1, int((st.session_state.portfolio["cash"] * 0.1) / base_price))
            
            if quantity > 0:
                price = base_price * (1 + random.uniform(-0.02, 0.02))
                value = quantity * price
                
                # Add the position
                positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": price,
                    "current_price": price,
                    "current_value": value,
                    "unrealized_pnl": 0,
                    "unrealized_pnl_pct": 0
                }
                
                # Deduct from cash
                st.session_state.portfolio["cash"] -= value
                
                # Log the trade
                st.session_state.logs.append(f"{timestamp} - BUY {symbol}: {quantity} shares @ ${price:.2f}")
    
    elif action == "SELL" and positions:
        # Sell existing position (partial or full)
        symbol = random.choice(list(positions.keys()))
        position = positions[symbol]
        
        # Determine if full or partial sale
        is_full_sale = random.random() > 0.7
        
        if is_full_sale:
            # Sell entire position
            sale_value = position["current_value"]
            st.session_state.portfolio["cash"] += sale_value
            
            # Log the trade
            st.session_state.logs.append(f"{timestamp} - SELL {symbol}: Closed position ({position['quantity']} shares) @ ${position['current_price']:.2f}")
            
            # Remove the position
            del positions[symbol]
        else:
            # Partial sale
            sell_quantity = max(1, int(position["quantity"] * random.uniform(0.1, 0.5)))
            
            if sell_quantity >= position["quantity"]:
                # If we're selling everything, treat as full sale
                sale_value = position["current_value"]
                st.session_state.portfolio["cash"] += sale_value
                
                # Log the trade
                st.session_state.logs.append(f"{timestamp} - SELL {symbol}: Closed position ({position['quantity']} shares) @ ${position['current_price']:.2f}")
                
                # Remove the position
                del positions[symbol]
            else:
                # Partial sale
                sale_value = sell_quantity * position["current_price"]
                remaining_quantity = position["quantity"] - sell_quantity
                
                # Update position
                position["quantity"] = remaining_quantity
                position["current_value"] = remaining_quantity * position["current_price"]
                position["unrealized_pnl"] = position["current_value"] - (remaining_quantity * position["avg_price"])
                position["unrealized_pnl_pct"] = (position["unrealized_pnl"] / (remaining_quantity * position["avg_price"])) * 100
                
                # Add to cash
                st.session_state.portfolio["cash"] += sale_value
                
                # Log the trade
                st.session_state.logs.append(f"{timestamp} - SELL {symbol}: {sell_quantity} shares @ ${position['current_price']:.2f}")

# Auto-update the portfolio while active
if st.session_state.trading_active:
    update_portfolio_data()
