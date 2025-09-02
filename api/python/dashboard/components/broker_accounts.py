"""
Broker Accounts Component

Displays detailed information about all connected broker accounts,
including balance, equity, and performance metrics for each broker.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_broker_accounts(data_service):
    """
    Render the broker accounts component, showing all connected brokers
    with detailed account information.
    
    Args:
        data_service: Data service for fetching broker data
    """
    # Get all broker accounts
    broker_df = data_service.get_broker_balances()
    
    # Display connection status at the top
    if hasattr(data_service, "is_using_mock_data"):
        connection_status = not data_service.is_using_mock_data()
    else:
        connection_status = False
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Broker Accounts")
    with col2:
        if connection_status:
            st.success("✅ Connected to Live API")
        else:
            st.warning("⚠️ Using Mock Data")
            if st.button("Retry Connection"):
                # Force refresh connection
                if hasattr(data_service, "_check_connection"):
                    data_service._check_connection()
                st.rerun()
    
    # Check if we have broker data
    if broker_df.empty:
        st.warning("No broker accounts configured. Please connect your trading accounts.")
        # Show configuration instructions
        with st.expander("How to connect brokers"):
            st.markdown("""
            ### Connecting Your Brokers
            
            To connect a broker, you need to:
            
            1. **Generate API keys** from your broker's website
            2. **Configure the API keys** in your config.py file
            3. **Restart the API server** to load the new configuration
            
            Supported brokers include:
            - Alpaca
            - Interactive Brokers
            - Binance
            - TD Ameritrade
            - Many more via the broker integration system
            """)
        return
    
    # Filter by account type if needed
    account_type = st.session_state.get("account_type", "Live")
    if account_type != "All":
        filtered_df = broker_df[broker_df['type'] == account_type]
    else:
        filtered_df = broker_df
    
    # Display broker accounts in a professional grid
    broker_count = len(filtered_df)
    if broker_count == 0:
        st.info(f"No {account_type} broker accounts configured.")
        return
    
    # Create grid layout based on number of brokers
    cols_per_row = min(3, broker_count)
    cols = st.columns(cols_per_row)
    
    # Display each broker as a card
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        col_idx = i % cols_per_row
        with cols[col_idx]:
            # Create styled broker card
            daily_pnl = row.get('daily_pnl', 0)
            daily_pnl_pct = row.get('daily_pnl_pct', 0)
            
            pnl_color = "#4CAF50" if daily_pnl >= 0 else "#F44336"
            
            # Card with broker logo, balances and performance
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: white; 
                        border-left: 5px solid {pnl_color}; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; color: black; font-weight: 600;">{row['name']}</h3>
                    <span style="background-color: {'#4CAF50' if row['status']=='Active' else '#FFA726'}; 
                           color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem;">
                        {row['status']}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #555;">Account Type:</span>
                    <span style="color: black; font-weight: 500;">{row['type']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #555;">Equity:</span>
                    <span style="color: black; font-weight: 500;">${row['equity']:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #555;">Cash:</span>
                    <span style="color: black; font-weight: 500;">${row['cash']:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #555;">Buying Power:</span>
                    <span style="color: black; font-weight: 500;">${row.get('buying_power', 0):,.2f}</span>
                </div>
                <div style="height: 1px; background-color: #eee; margin: 10px 0;"></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0;">
                    <span style="color: #555;">Today P&L:</span>
                    <span style="color: {pnl_color}; font-weight: 600;">
                        ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add action buttons for the broker
            col1, col2 = st.columns(2)
            with col1:
                st.button("Details", key=f"details_{row['broker_id']}")
            with col2:
                st.button("Settings", key=f"settings_{row['broker_id']}")
    
    # Show total portfolio value across all accounts
    total_equity = filtered_df['equity'].sum()
    total_cash = filtered_df['cash'].sum()
    total_pnl = filtered_df['daily_pnl'].sum()
    total_pnl_pct = total_pnl / (total_equity - total_pnl) * 100 if (total_equity - total_pnl) != 0 else 0
    
    st.markdown("---")
    
    # Create portfolio summary columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Portfolio Value",
            value=f"${total_equity:,.2f}",
            delta=None
        )
    with col2:
        st.metric(
            label="Available Cash",
            value=f"${total_cash:,.2f}",
            delta=None
        )
    with col3:
        st.metric(
            label="Today's P&L",
            value=f"${total_pnl:,.2f}",
            delta=f"{total_pnl_pct:.2f}%",
            delta_color="normal"
        )
    with col4:
        # Get positions count
        positions_df = data_service.get_positions()
        position_count = len(positions_df) if not positions_df.empty else 0
        
        st.metric(
            label="Active Positions",
            value=position_count,
            delta=None
        )
    
    # Display breakdown by asset class if positions are available
    if not positions_df.empty and 'asset_class' in positions_df.columns:
        st.subheader("Portfolio Allocation")
        
        # Group by asset class
        asset_breakdown = positions_df.groupby('asset_class')['market_value'].sum().reset_index()
        total_value = asset_breakdown['market_value'].sum()
        asset_breakdown['percentage'] = asset_breakdown['market_value'] / total_value * 100
        
        # Create pie chart
        fig = px.pie(
            asset_breakdown, 
            values='percentage', 
            names='asset_class',
            title='Allocation by Asset Class',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=30, b=10, l=10, r=10),
            font=dict(color="black")
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_connection_manager(data_service):
    """
    Render the broker connection manager to add and manage broker connections.
    
    Args:
        data_service: Data service for broker operations
    """
    st.header("Broker Connection Manager")
    
    # Show existing connections
    broker_df = data_service.get_broker_balances()
    
    if not broker_df.empty:
        st.subheader("Connected Brokers")
        for _, row in broker_df.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{row['name']}** ({row['type']})")
            with col2:
                st.write(f"Status: {row['status']}")
            with col3:
                st.button("Disconnect", key=f"disconnect_{row['broker_id']}")
    
    # Add new broker connection
    st.subheader("Add New Broker")
    
    col1, col2 = st.columns(2)
    with col1:
        broker_type = st.selectbox(
            "Broker Type",
            options=["Alpaca", "Interactive Brokers", "TD Ameritrade", "Binance", "Coinbase", "Other"]
        )
    with col2:
        account_type = st.selectbox(
            "Account Type",
            options=["Live", "Paper"]
        )
    
    # API key fields
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    
    # Additional fields based on broker type
    if broker_type == "Interactive Brokers":
        st.text_input("Account ID")
    elif broker_type in ["Binance", "Coinbase"]:
        st.selectbox("Base Currency", options=["USD", "EUR", "GBP", "JPY"])
    
    # Submit button
    if st.button("Connect Broker"):
        st.success(f"Successfully connected to {broker_type}!")
        # In real implementation, this would call the API service to save credentials
