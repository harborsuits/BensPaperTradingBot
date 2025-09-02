#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading Dashboard...${NC}"

# Set directories
DASHBOARD_DIR="/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"
PARENT_DIR="/Users/bendickinson/Desktop/Trading:BenBot"

# Activate the virtual environment
source ~/bensbot_env/bin/activate

# Check if Streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo -e "${YELLOW}Streamlit not found. Installing required packages...${NC}"
    pip install streamlit pandas numpy matplotlib plotly
fi

# Create a simplified dashboard file that will work with our setup
cat > "$DASHBOARD_DIR/simplified_dashboard.py" << 'EOF'
"""
Simplified Dashboard for BensBot Paper Trading
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pymongo
import plotly.graph_objects as go
import random
import yfinance as yf

# Setup page config
st.set_page_config(
    page_title="BensBot Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #14375F;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MongoDB connection
try:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/bensbot_trading")
    mongo_client = pymongo.MongoClient(mongo_uri)
    db = mongo_client.get_database()
    mongo_available = True
    st.sidebar.success("Connected to MongoDB")
except Exception as e:
    mongo_available = False
    st.sidebar.error(f"MongoDB connection error: {e}")

# Fetch real-time market data for a symbol
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_market_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if len(data) > 0:
            return data.iloc[-1]['Close']
        return None
    except Exception as e:
        st.warning(f"Error fetching price for {symbol}: {e}")
        return None

# Generate mock data when MongoDB isn't available
def generate_mock_data():
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    positions = []
    
    # Generate positions
    for symbol in symbols:
        if random.random() < 0.7:  # 70% chance to have a position
            price = get_market_price(symbol) or random.uniform(100, 500)
            quantity = round(random.uniform(5, 100), 2)
            avg_price = price * random.uniform(0.8, 1.2)  # Random entry price around current
            market_value = price * quantity
            unrealized_pnl = market_value - (avg_price * quantity)
            
            positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl
            })
    
    # Generate orders
    orders = []
    for _ in range(10):
        symbol = random.choice(symbols)
        side = random.choice(["buy", "sell"])
        status = random.choice(["filled", "open", "cancelled"])
        price = get_market_price(symbol) or random.uniform(100, 500)
        quantity = round(random.uniform(1, 50), 2)
        
        # Random date within last 7 days
        days_ago = random.uniform(0, 7)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        orders.append({
            "order_id": f"mock_{random.randint(1000, 9999)}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": status,
            "timestamp": timestamp
        })
    
    # Sort by timestamp (newest first)
    orders.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Generate account info
    account = {
        "balance": random.uniform(80000, 120000),
        "securities_value": sum(p["market_value"] for p in positions),
        "starting_balance": 100000
    }
    account["total_equity"] = account["balance"] + account["securities_value"]
    account["total_pnl"] = account["total_equity"] - account["starting_balance"]
    account["pnl_pct"] = (account["total_pnl"] / account["starting_balance"]) * 100
    
    return {"positions": positions, "orders": orders, "account": account}

# Get portfolio data
def get_portfolio_data():
    if mongo_available:
        try:
            # Get positions from MongoDB
            positions = list(db.paper_positions.find({"account_type": "paper"}))
            
            # Update current prices and calculations
            for position in positions:
                symbol = position.get("symbol")
                if symbol:
                    current_price = get_market_price(symbol)
                    if current_price:
                        quantity = position.get("quantity", 0)
                        position["current_price"] = current_price
                        position["market_value"] = quantity * current_price
                        position["unrealized_pnl"] = position["market_value"] - (quantity * position.get("avg_price", 0))
            
            # Get orders
            orders = list(db.paper_orders.find({"account_type": "paper"}).sort("created_at", -1).limit(20))
            
            # Get account info
            account_doc = db.paper_account.find_one({"account_type": "paper"})
            if account_doc:
                account = {
                    "balance": account_doc.get("balance", 0),
                    "securities_value": sum(p.get("market_value", 0) for p in positions),
                    "starting_balance": 100000  # Default starting balance
                }
                account["total_equity"] = account["balance"] + account["securities_value"]
                account["total_pnl"] = account["total_equity"] - account["starting_balance"]
                account["pnl_pct"] = (account["total_pnl"] / account["starting_balance"]) * 100
            else:
                account = {"balance": 0, "securities_value": 0, "total_equity": 0, 
                           "total_pnl": 0, "pnl_pct": 0, "starting_balance": 100000}
            
            return {"positions": positions, "orders": orders, "account": account}
        except Exception as e:
            st.error(f"Error fetching data from MongoDB: {e}")
            return generate_mock_data()
    else:
        return generate_mock_data()

# Sidebar
st.sidebar.title("BensBot Trading")
st.sidebar.image("https://img.icons8.com/fluency/96/000000/exchange.png", width=80)

# Account type selector
account_type = st.sidebar.selectbox(
    "Account Type",
    ["Paper Trading", "Live Trading"],
    index=0
)

if account_type == "Live Trading":
    st.sidebar.warning("Live trading is not available in this simplified dashboard.")

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if auto_refresh:
    st.sidebar.caption("Dashboard will refresh every 30 seconds")
    time.sleep(1)
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

# Main content
st.title("Trading Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Portfolio Summary", "Positions", "Order History"])

# Get all data
data = get_portfolio_data()
positions = data["positions"]
orders = data["orders"]
account = data["account"]

# Tab 1: Portfolio Summary
with tab1:
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Equity", 
            f"${account['total_equity']:,.2f}",
            f"{account['pnl_pct']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Cash Balance", 
            f"${account['balance']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Securities Value", 
            f"${account['securities_value']:,.2f}"
        )
    
    with col4:
        st.metric(
            "Total P&L", 
            f"${account['total_pnl']:,.2f}",
            f"{account['pnl_pct']:.2f}%"
        )
    
    # Portfolio composition pie chart
    if positions:
        portfolio_data = []
        labels = []
        values = []
        
        for pos in positions:
            if pos.get("market_value", 0) > 0:
                labels.append(pos.get("symbol", "Unknown"))
                values.append(pos.get("market_value", 0))
        
        if values:
            # Add cash as a position
            labels.append("Cash")
            values.append(account["balance"])
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                textinfo='label+percent',
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No positions found. Your portfolio is 100% cash.")

# Tab 2: Positions
with tab2:
    if positions:
        # Convert to DataFrame for easier display
        df_positions = pd.DataFrame(positions)
        
        # Select relevant columns and rename them
        cols_to_display = [
            "symbol", "quantity", "avg_price", "current_price", 
            "market_value", "unrealized_pnl"
        ]
        
        col_names = {
            "symbol": "Symbol",
            "quantity": "Quantity",
            "avg_price": "Avg. Price",
            "current_price": "Current Price",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L"
        }
        
        # Format the DataFrame
        if all(col in df_positions.columns for col in cols_to_display):
            df_display = df_positions[cols_to_display].rename(columns=col_names)
            
            # Apply formatting
            df_display["Avg. Price"] = df_display["Avg. Price"].apply(lambda x: f"${x:,.2f}")
            df_display["Current Price"] = df_display["Current Price"].apply(lambda x: f"${x:,.2f}")
            df_display["Market Value"] = df_display["Market Value"].apply(lambda x: f"${x:,.2f}")
            
            # Apply conditional formatting to P&L
            def format_pnl(val):
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f"<span style='color:{color}'>${val:,.2f}</span>"
            
            df_display["Unrealized P&L"] = df_positions["unrealized_pnl"].apply(format_pnl)
            
            # Display the table
            st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.error("Position data format is incorrect")
    else:
        st.info("No open positions found.")

# Tab 3: Order History
with tab3:
    if orders:
        # Convert to DataFrame for easier display
        df_orders = pd.DataFrame(orders)
        
        # Select relevant columns and rename them
        cols_to_display = [
            "order_id", "symbol", "side", "quantity", "price", 
            "status", "timestamp"
        ]
        
        col_names = {
            "order_id": "Order ID",
            "symbol": "Symbol",
            "side": "Side",
            "quantity": "Quantity",
            "price": "Price",
            "status": "Status",
            "timestamp": "Timestamp"
        }
        
        # Check columns exist
        if all(col in df_orders.columns for col in cols_to_display):
            df_display = df_orders[cols_to_display].rename(columns=col_names)
            
            # Apply formatting
            df_display["Side"] = df_display["Side"].str.upper()
            df_display["Price"] = df_display["Price"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            
            # Format status
            def format_status(status):
                color = {"filled": "green", "open": "blue", "cancelled": "orange"}.get(status.lower(), "gray")
                return f"<span style='color:{color}'>{status.upper()}</span>"
            
            df_display["Status"] = df_orders["status"].apply(format_status)
            
            # Format timestamps
            if "Timestamp" in df_display.columns:
                df_display["Timestamp"] = pd.to_datetime(df_display["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Display the table
            st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            missing = [col for col in cols_to_display if col not in df_orders.columns]
            st.error(f"Order data missing columns: {missing}")
    else:
        st.info("No orders found.")

st.caption("BensBot Trading Dashboard - Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
EOF

echo -e "${GREEN}Dashboard files ready.${NC}"

# Navigate to the dashboard directory
cd "$DASHBOARD_DIR"

# Start the dashboard
echo -e "${YELLOW}Starting Streamlit dashboard...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard when done.${NC}"

# Start the Streamlit server
PYTHONPATH="$PARENT_DIR" streamlit run simplified_dashboard.py --server.port=8501

echo -e "${YELLOW}Dashboard stopped.${NC}"
