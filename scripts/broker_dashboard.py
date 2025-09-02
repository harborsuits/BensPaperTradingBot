"""
BensBot Multi-Broker Trading Dashboard

A simplified standalone dashboard showing all broker accounts with real data connection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime
import time
import os
import sys

# Set page config
st.set_page_config(
    page_title="BensBot Multi-Broker Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling 
st.markdown("""
<style>
    /* Clean, high-contrast styling */
    .main { background-color: #f5f5f5; color: black; }
    .stSidebar { background-color: #222; }
    .stSidebar [data-testid="stMarkdownContainer"] p { color: white !important; }
    .stSidebar [data-testid="stMarkdownContainer"] h1, h2, h3 { color: white !important; }
    
    /* Cards for broker accounts */
    .broker-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .positive { color: #4CAF50; font-weight: bold; }
    .negative { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class DataService:
    """Service to fetch data from the trading API or use mock data."""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.is_connected = False
        self.using_mock_data = True
        self.connection_attempts = 0
        
        # Try to connect to the API
        self.check_connection()
        
    def check_connection(self):
        """Check connection to the backend API."""
        try:
            self.connection_attempts += 1
            response = requests.get(f"{self.api_url}/", timeout=3)
            if response.status_code == 200:
                self.is_connected = True
                self.using_mock_data = False
                return True
        except:
            self.is_connected = False
            self.using_mock_data = True
            return False
        return False

    def get_broker_accounts(self):
        """Get all broker accounts."""
        if not self.using_mock_data:
            try:
                response = requests.get(f"{self.api_url}/api/portfolio/accounts", timeout=3)
                if response.status_code == 200:
                    return pd.DataFrame(response.json().get("accounts", []))
            except:
                pass
        
        # Return mock data if API failed or we're in mock mode
        return pd.DataFrame([
            {
                "broker_id": "alpaca_001",
                "name": "Alpaca",
                "type": "Live",
                "status": "Active",
                "cash": 25432.18,
                "equity": 32876.45,
                "buying_power": 64654.90,
                "daily_pnl": 245.87,
                "daily_pnl_pct": 0.76
            },
            {
                "broker_id": "interactive_brokers_001",
                "name": "Interactive Brokers",
                "type": "Live",
                "status": "Active",
                "cash": 105672.35,
                "equity": 189453.78,
                "buying_power": 211344.70,
                "daily_pnl": -342.15,
                "daily_pnl_pct": -0.18
            },
            {
                "broker_id": "binance_001",
                "name": "Binance",
                "type": "Live",
                "status": "Active",
                "cash": 5243.67,
                "equity": 15870.22,
                "buying_power": 5243.67,
                "daily_pnl": 876.32,
                "daily_pnl_pct": 5.84
            },
            {
                "broker_id": "paper_account_001",
                "name": "Paper Trading",
                "type": "Paper",
                "status": "Active",
                "cash": 100000.00,
                "equity": 103567.89,
                "buying_power": 200000.00,
                "daily_pnl": 567.89,
                "daily_pnl_pct": 0.55
            }
        ])
        
    def get_positions(self, broker_id=None):
        """Get positions for specified broker or all brokers."""
        if not self.using_mock_data:
            try:
                url = f"{self.api_url}/api/portfolio/positions"
                if broker_id:
                    url += f"?broker_id={broker_id}"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    return pd.DataFrame(response.json().get("positions", []))
            except:
                pass
        
        # Return mock data
        return pd.DataFrame([
            {
                "broker_id": "alpaca_001",
                "symbol": "AAPL",
                "quantity": 25,
                "entry_price": 175.23,
                "current_price": 182.45,
                "market_value": 4561.25,
                "unrealized_pnl": 180.50,
                "unrealized_pnl_pct": 4.13,
                "asset_class": "Stocks"
            },
            {
                "broker_id": "alpaca_001",
                "symbol": "MSFT",
                "quantity": 15,
                "entry_price": 322.45,
                "current_price": 328.78,
                "market_value": 4931.70,
                "unrealized_pnl": 94.95,
                "unrealized_pnl_pct": 1.97,
                "asset_class": "Stocks"
            },
            {
                "broker_id": "interactive_brokers_001",
                "symbol": "SPY",
                "quantity": 40,
                "entry_price": 450.32,
                "current_price": 458.75,
                "market_value": 18350.00,
                "unrealized_pnl": 337.20,
                "unrealized_pnl_pct": 1.87,
                "asset_class": "ETFs"
            },
            {
                "broker_id": "interactive_brokers_001",
                "symbol": "QQQ",
                "quantity": 30,
                "entry_price": 375.67,
                "current_price": 389.23,
                "market_value": 11676.90,
                "unrealized_pnl": 406.80,
                "unrealized_pnl_pct": 3.61,
                "asset_class": "ETFs"
            },
            {
                "broker_id": "binance_001",
                "symbol": "BTC/USD",
                "quantity": 0.25,
                "entry_price": 58742.30,
                "current_price": 62150.75,
                "market_value": 15537.69,
                "unrealized_pnl": 852.11,
                "unrealized_pnl_pct": 5.81,
                "asset_class": "Crypto"
            }
        ])
    
    def get_strategies(self, broker_id=None):
        """Get active strategies."""
        if not self.using_mock_data:
            try:
                url = f"{self.api_url}/api/strategies/list"
                if broker_id:
                    url += f"?broker_id={broker_id}"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    return pd.DataFrame(response.json().get("strategies", []))
            except:
                pass
        
        # Return mock data
        return pd.DataFrame([
            {
                "id": "momentum_01",
                "name": "Momentum Strategy",
                "broker_id": "alpaca_001",
                "status": "Active",
                "allocation": 15000.00,
                "daily_pnl": 124.50,
                "daily_pnl_pct": 0.83,
                "win_rate": 68.5
            },
            {
                "id": "mean_reversion_01",
                "name": "Mean Reversion",
                "broker_id": "interactive_brokers_001",
                "status": "Active",
                "allocation": 35000.00,
                "daily_pnl": -156.25,
                "daily_pnl_pct": -0.45,
                "win_rate": 54.2
            },
            {
                "id": "crypto_trend_01",
                "name": "Crypto Trend",
                "broker_id": "binance_001",
                "status": "Active",
                "allocation": 10000.00,
                "daily_pnl": 732.45,
                "daily_pnl_pct": 7.32,
                "win_rate": 61.8
            }
        ])

# Initialize the data service
@st.cache_resource
def init_data_service():
    return DataService()

data_service = init_data_service()

# Sidebar
with st.sidebar:
    st.title("BensBot Trading")
    
    # Connection status
    if data_service.is_connected:
        st.success("✅ Connected to Live API")
    else:
        st.warning("⚠️ Using Mock Data")
        if st.button("Connect to API"):
            data_service.check_connection()
            st.rerun()
    
    # Account type filter
    st.header("Account Filter")
    account_type = st.radio(
        "Account Type",
        options=["All", "Live", "Paper"],
        index=0
    )
    
    # Space for account metrics
    st.header("Portfolio Summary")
    all_accounts = data_service.get_broker_accounts()
    
    # Filter by account type if needed
    if account_type != "All":
        filtered_accounts = all_accounts[all_accounts['type'] == account_type]
    else:
        filtered_accounts = all_accounts
    
    # Calculate total values
    total_equity = filtered_accounts['equity'].sum()
    total_pnl = filtered_accounts['daily_pnl'].sum()
    
    # Display total portfolio value
    st.metric(
        "Total Portfolio Value",
        f"${total_equity:,.2f}",
        f"{total_pnl:+,.2f}" if total_pnl != 0 else None,
        delta_color="normal" if total_pnl >= 0 else "inverse"
    )

# Main area
st.title("Multi-Broker Dashboard")

# Data source indicator with refresh button
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
with col2:
    if st.button("Refresh Data"):
        st.rerun()

# Get the broker accounts
all_accounts = data_service.get_broker_accounts()

# Filter by account type if needed
if account_type != "All":
    filtered_accounts = all_accounts[all_accounts['type'] == account_type]
else:
    filtered_accounts = all_accounts

# Broker accounts section
st.header(f"{account_type} Broker Accounts")

# Create grid layout for broker cards
num_accounts = len(filtered_accounts)
if num_accounts == 0:
    st.info(f"No {account_type} accounts configured.")
else:
    # Determine number of columns (max 3)
    cols_per_row = min(3, num_accounts)
    cols = st.columns(cols_per_row)
    
    # Display broker cards
    for i, (_, account) in enumerate(filtered_accounts.iterrows()):
        col_idx = i % cols_per_row
        with cols[col_idx]:
            # Format broker card
            pnl_color = "positive" if account['daily_pnl'] >= 0 else "negative"
            
            st.markdown(f"""
            <div class="broker-card" style="border-left: 5px solid {'green' if account['daily_pnl'] >= 0 else 'red'};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; color: black;">{account['name']}</h3>
                    <span style="background-color: {'green' if account['status'] == 'Active' else 'orange'}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem;">
                        {account['status']}
                    </span>
                </div>
                <p style="margin: 5px 0; color: #555;">Account Type: <strong style="color: black;">{account['type']}</strong></p>
                <p style="margin: 5px 0; color: #555;">Equity: <strong style="color: black;">${account['equity']:,.2f}</strong></p>
                <p style="margin: 5px 0; color: #555;">Cash: <strong style="color: black;">${account['cash']:,.2f}</strong></p>
                <p style="margin: 5px 0; color: #555;">Today's P&L: <strong class="{pnl_color}">${account['daily_pnl']:,.2f} ({account['daily_pnl_pct']:.2f}%)</strong></p>
                <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                    <span style="background-color: #007bff; color: white; padding: 5px 15px; border-radius: 5px; cursor: pointer; text-align: center; width: 48%;">Positions</span>
                    <span style="background-color: #6c757d; color: white; padding: 5px 15px; border-radius: 5px; cursor: pointer; text-align: center; width: 48%;">Settings</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Asset Allocation tab
st.header("Asset Allocation")

# Get positions
all_positions = data_service.get_positions()

# Filter positions by broker if needed
if account_type != "All":
    filtered_brokers = filtered_accounts['broker_id'].tolist()
    filtered_positions = all_positions[all_positions['broker_id'].isin(filtered_brokers)]
else:
    filtered_positions = all_positions

if filtered_positions.empty:
    st.info("No positions found for the selected account type.")
else:
    # Create allocation pie chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Group by asset class
        asset_allocation = filtered_positions.groupby('asset_class')['market_value'].sum().reset_index()
        total_value = asset_allocation['market_value'].sum()
        asset_allocation['percentage'] = asset_allocation['market_value'] / total_value * 100
        
        # Create pie chart
        fig = px.pie(
            asset_allocation, 
            values='percentage', 
            names='asset_class',
            title='Portfolio Allocation by Asset Class',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=30, b=0, l=10, r=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show allocation table
        st.subheader("Allocation Breakdown")
        
        # Format the allocation table
        allocation_table = asset_allocation.copy()
        allocation_table['market_value'] = allocation_table['market_value'].apply(lambda x: f"${x:,.2f}")
        allocation_table['percentage'] = allocation_table['percentage'].apply(lambda x: f"{x:.2f}%")
        allocation_table.columns = ['Asset Class', 'Value', 'Percentage']
        
        st.table(allocation_table)

# Active Strategies section
st.header("Active Trading Strategies")

# Get strategies
all_strategies = data_service.get_strategies()

# Filter strategies if needed
if account_type != "All":
    filtered_brokers = filtered_accounts['broker_id'].tolist()
    filtered_strategies = all_strategies[all_strategies['broker_id'].isin(filtered_brokers)]
else:
    filtered_strategies = all_strategies

if filtered_strategies.empty:
    st.info("No active strategies found for the selected account type.")
else:
    # Display strategies in a table with styling
    for _, strategy in filtered_strategies.iterrows():
        # Get broker name
        broker_name = filtered_accounts[filtered_accounts['broker_id'] == strategy['broker_id']]['name'].values[0] if len(filtered_accounts) > 0 else "Unknown"
        
        # Format P&L
        pnl_class = "positive" if strategy['daily_pnl'] >= 0 else "negative"
        
        # Create strategy card
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; color: black;">{strategy['name']}</h3>
                <span style="background-color: {'green' if strategy['status'] == 'Active' else 'orange'}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem;">
                    {strategy['status']}
                </span>
            </div>
            <p style="margin: 5px 0; color: #555;">Broker: <strong style="color: black;">{broker_name}</strong></p>
            <p style="margin: 5px 0; color: #555;">Allocation: <strong style="color: black;">${strategy['allocation']:,.2f}</strong></p>
            <p style="margin: 5px 0; color: #555;">Today's P&L: <strong class="{pnl_class}">${strategy['daily_pnl']:,.2f} ({strategy['daily_pnl_pct']:.2f}%)</strong></p>
            <p style="margin: 5px 0; color: #555;">Win Rate: <strong style="color: black;">{strategy['win_rate']:.1f}%</strong></p>
            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                <span style="background-color: #28a745; color: white; padding: 5px 15px; border-radius: 5px; cursor: pointer; text-align: center; width: 30%;">Performance</span>
                <span style="background-color: #ffc107; color: black; padding: 5px 15px; border-radius: 5px; cursor: pointer; text-align: center; width: 30%;">Modify</span>
                <span style="background-color: #dc3545; color: white; padding: 5px 15px; border-radius: 5px; cursor: pointer; text-align: center; width: 30%;">Stop</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Connection information at bottom
st.markdown("---")
st.caption(f"API Connection Status: {'Connected to Live API' if data_service.is_connected else 'Using Mock Data'}")
st.caption(f"Connection Attempts: {data_service.connection_attempts}")
