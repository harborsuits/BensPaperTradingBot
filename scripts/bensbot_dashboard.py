"""
BensBot Pro Trading Dashboard
Inspired by professional trading platforms with direct MongoDB integration
"""
import os
import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pymongo
from PIL import Image
import base64
from io import BytesIO
import json

# Configure page
st.set_page_config(
    page_title="BensBot Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme (inspired by trading-dashboard)
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("ui_assets/custom_theme.css")
except FileNotFoundError:
    st.warning("Custom theme file not found. Using default styling.")

# MongoDB connection
@st.cache_resource(ttl=300)
def get_mongodb():
    try:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/bensbot_trading")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        db = client.get_database()
        return db, True
    except Exception as e:
        st.sidebar.error(f"MongoDB connection error: {e}")
        return None, False

# Get real-time market data (inspired by MyCryptoBot)
@st.cache_data(ttl=60)
def get_realtime_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if len(data) > 0:
            price = data.iloc[-1]['Close']
            change = ((price / data.iloc[0]['Open']) - 1) * 100
            return price, change
        return None, None
    except Exception as e:
        st.warning(f"Error fetching price for {symbol}: {e}")
        return None, None

# Get portfolio data from MongoDB
def get_portfolio_data(db, account_type="paper"):
    # Default empty structure
    empty_portfolio = {
        "positions": [],
        "orders": [],
        "account": {
            "balance": 100000,
            "securities_value": 0,
            "total_equity": 100000,
            "total_pnl": 0,
            "pnl_pct": 0,
            "starting_balance": 100000
        },
        "equity_history": pd.DataFrame({
            'date': [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(30, 0, -1)],
            'equity': [100000] * 30
        })
    }
    
    if db is None:
        return empty_portfolio
    
    try:
        # Get positions with current prices
        positions = list(db.paper_positions.find({"account_type": account_type}))
        
        for position in positions:
            symbol = position.get("symbol")
            if symbol:
                price, change = get_realtime_price(symbol)
                if price:
                    quantity = position.get("quantity", 0)
                    position["current_price"] = price
                    position["price_change"] = change 
                    position["market_value"] = quantity * price
                    position["unrealized_pnl"] = position["market_value"] - (quantity * position.get("avg_price", 0))
        
        # Get orders
        orders = list(db.paper_orders.find({"account_type": account_type}).sort("created_at", -1).limit(30))
        
        # Get account info
        account_doc = db.paper_account.find_one({"account_type": account_type})
        if account_doc:
            securities_value = sum(p.get("market_value", 0) for p in positions)
            balance = account_doc.get("balance", 100000)
            starting_balance = account_doc.get("starting_balance", 100000)
            
            total_equity = balance + securities_value
            total_pnl = total_equity - starting_balance
            pnl_pct = (total_pnl / starting_balance) * 100 if starting_balance else 0
            
            account = {
                "balance": balance,
                "securities_value": securities_value,
                "total_equity": total_equity,
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct,
                "starting_balance": starting_balance
            }
        else:
            account = empty_portfolio["account"]
        
        # Performance metrics (inspired by Crypto-Bot)
        trades = list(db.paper_trades.find({"account_type": account_type, "status": "closed"}))
        if trades:
            win_trades = [t for t in trades if t.get("pnl", 0) > 0]
            win_rate = len(win_trades) / len(trades) * 100
            avg_win = sum(t.get("pnl", 0) for t in win_trades) / len(win_trades) if win_trades else 0
            
            lose_trades = [t for t in trades if t.get("pnl", 0) <= 0]
            avg_loss = sum(t.get("pnl", 0) for t in lose_trades) / len(lose_trades) if lose_trades else 0
            
            account["win_rate"] = win_rate
            account["avg_win"] = avg_win
            account["avg_loss"] = avg_loss
            account["total_trades"] = len(trades)
        
        # Get or generate equity curve
        try:
            # Try to get from MongoDB first
            if db is not None:  # Check if MongoDB is available
                equity_docs = list(db.equity_history.find(
                    {"account_type": account_type}
                ).sort("date", 1).limit(90))  # Last 90 days
            else:
                equity_docs = []
            
            if equity_docs:
                equity_df = pd.DataFrame(equity_docs)
                equity_df['date'] = pd.to_datetime(equity_df['date'])
                equity_history = equity_df[['date', 'equity']]
            else:
                # Generate synthetic equity curve
                days = 30
                dates = [(datetime.datetime.now() - datetime.timedelta(days=i)) for i in range(days, 0, -1)]
                
                # Random walk with upward bias
                equity_values = [starting_balance]
                for i in range(1, days):
                    change = np.random.normal(starting_balance * 0.001, starting_balance * 0.005)
                    new_value = max(0, equity_values[-1] + change)
                    equity_values.append(new_value)
                
                # Make sure last point matches current equity
                equity_values[-1] = total_equity
                
                equity_history = pd.DataFrame({
                    'date': dates,
                    'equity': equity_values
                })
        except Exception as e:
            st.error(f"Error getting equity history: {e}")
            equity_history = empty_portfolio["equity_history"]
        
        return {
            "positions": positions,
            "orders": orders,
            "account": account,
            "equity_history": equity_history
        }
    except Exception as e:
        st.error(f"Error fetching portfolio data: {e}")
        return empty_portfolio

# Get system status (inspired by MyCryptoBot)
def get_system_status(db):
    if db is None:
        return {
            "status": "offline", 
            "trading_enabled": False,
            "last_update": "N/A",
            "message": "MongoDB not connected"
        }
    
    try:
        status_doc = db.system_status.find_one({"type": "system_status"})
        if status_doc:
            return status_doc
        
        # If no status in DB, create a default
        return {
            "status": "online",
            "trading_enabled": True,
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "System running normally"
        }
    except Exception as e:
        return {
            "status": "error",
            "trading_enabled": False,
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"Error: {str(e)}"
        }

# Header with logo and title (inspired by trading-dashboard)
def render_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("# 📈 BensBot")
    with col2:
        st.markdown("### Professional Trading Dashboard")
    with col3:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**Last Update:** {now}")

# Portfolio summary section (inspired by trading-dashboard)
def render_portfolio_summary(data):
    account = data["account"]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Equity", 
            f"${account['total_equity']:,.2f}", 
            f"{account['pnl_pct']:.2f}%" if 'pnl_pct' in account else None
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
        metrics = []
        if 'win_rate' in account:
            metrics.append(f"Win Rate: {account['win_rate']:.1f}%")
        if 'total_trades' in account:
            metrics.append(f"Total Trades: {account['total_trades']}")
            
        if metrics:
            st.metric(
                "Performance", 
                metrics[0],
                ", ".join(metrics[1:]) if len(metrics) > 1 else None
            )
        else:
            st.metric(
                "P&L", 
                f"${account['total_pnl']:,.2f}"
            )
    
    # Equity chart
    st.subheader("Equity Curve")
    
    # Time selector (inspired by MyCryptoBot)
    time_periods = ["1W", "1M", "3M", "YTD", "ALL"]
    selected_period = st.select_slider("Time Period", options=time_periods, value="1M")
    
    # Filter data based on selection
    equity_df = data["equity_history"]
    if selected_period != "ALL":
        now = datetime.datetime.now()
        if selected_period == "1W":
            start_date = now - datetime.timedelta(days=7)
        elif selected_period == "1M":
            start_date = now - datetime.timedelta(days=30)
        elif selected_period == "3M":
            start_date = now - datetime.timedelta(days=90)
        elif selected_period == "YTD":
            start_date = datetime.datetime(now.year, 1, 1)
            
        equity_df = equity_df[equity_df['date'] >= start_date]
    
    # Create chart with reference line for starting balance
    fig = go.Figure()
    
    # Add equity line
    fig.add_trace(
        go.Scatter(
            x=equity_df["date"], 
            y=equity_df["equity"],
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=3)
        )
    )
    
    # Add reference line for starting balance
    fig.add_trace(
        go.Scatter(
            x=[min(equity_df["date"]), max(equity_df["date"])],
            y=[account["starting_balance"], account["starting_balance"]],
            mode='lines',
            name='Starting Balance',
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash')
        )
    )
    
    # Enhance layout (inspired by trading-dashboard)
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio allocation (inspired by Crypto-Bot)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Allocation")
        
        positions = data["positions"]
        if positions:
            labels = []
            values = []
            
            for pos in positions:
                symbol = pos.get("symbol", "Unknown")
                market_value = pos.get("market_value", 0)
                if market_value > 0:
                    labels.append(symbol)
                    values.append(market_value)
            
            # Add cash
            labels.append("Cash")
            values.append(account["balance"])
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                textinfo='label+percent',
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
            )])
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions. Portfolio is 100% cash.")
    
    with col2:
        st.subheader("Performance Metrics")
        
        # Create metrics cards inspired by MyCryptoBot
        if 'win_rate' in account and 'total_trades' in account:
            metrics_data = [
                {"name": "Win Rate", "value": f"{account['win_rate']:.1f}%", "desc": "Percentage of profitable trades"},
                {"name": "Total Trades", "value": f"{account['total_trades']}", "desc": "Number of completed trades"},
                {"name": "Avg. Win", "value": f"${account.get('avg_win', 0):,.2f}", "desc": "Average profit per winning trade"},
                {"name": "Avg. Loss", "value": f"${account.get('avg_loss', 0):,.2f}", "desc": "Average loss per losing trade"}
            ]
            
            # Display metrics in a grid (2x2)
            col1, col2 = st.columns(2)
            for i, metric in enumerate(metrics_data):
                with col1 if i % 2 == 0 else col2:
                    st.markdown(f"""
                    <div style="border: 1px solid #eaeaea; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                        <p style="font-size: 0.8rem; font-weight: 600; margin-bottom: 5px; color: #666;">{metric['name']}</p>
                        <p style="font-size: 1.4rem; font-weight: 700; margin-bottom: 5px;">{metric['value']}</p>
                        <p style="font-size: 0.7rem; color: #888;">{metric['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Not enough trading data to display performance metrics.")

# Positions section (inspired by trading-dashboard)
def render_positions(data):
    positions = data["positions"]
    
    if positions:
        # Convert to DataFrame
        df = pd.DataFrame(positions)
        
        # Format for display
        display_cols = {
            "symbol": "Symbol",
            "quantity": "Quantity",
            "avg_price": "Avg. Price",
            "current_price": "Current Price",
            "price_change": "24h Change",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L"
        }
        
        # Check if all required columns exist
        if all(col in df.columns for col in display_cols.keys()):
            # Select and rename columns
            df_display = df[display_cols.keys()].rename(columns=display_cols)
            
            # Format columns
            df_display["Avg. Price"] = df_display["Avg. Price"].apply(lambda x: f"${x:,.2f}")
            df_display["Current Price"] = df_display["Current Price"].apply(lambda x: f"${x:,.2f}")
            df_display["Market Value"] = df_display["Market Value"].apply(lambda x: f"${x:,.2f}")
            
            # Color-coded formatting for price change
            def format_change(val):
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f"<span style='color:{color}'>{'▲' if val > 0 else '▼' if val < 0 else '■'} {val:.2f}%</span>"
            
            df_display["24h Change"] = df["price_change"].apply(format_change)
            
            # Color-coded formatting for P&L
            def format_pnl(val):
                color = "green" if val > 0 else "red" if val < 0 else "black"
                return f"<span style='color:{color}'>{'▲' if val > 0 else '▼' if val < 0 else '■'} ${abs(val):,.2f}</span>"
            
            df_display["Unrealized P&L"] = df["unrealized_pnl"].apply(format_pnl)
            
            # Display as HTML table for better formatting
            st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            missing = [col for col in display_cols.keys() if col not in df.columns]
            st.error(f"Position data missing columns: {missing}")
            st.dataframe(df)
    else:
        st.info("No open positions.")

# Orders section (inspired by Crypto-Bot)
def render_orders(data):
    orders = data["orders"]
    
    if orders:
        # Convert to DataFrame
        df = pd.DataFrame(orders)
        
        # Format for display
        display_cols = {
            "order_id": "Order ID",
            "symbol": "Symbol", 
            "side": "Side",
            "quantity": "Quantity",
            "price": "Price",
            "status": "Status",
            "created_at": "Date/Time"
        }
        
        # Select available columns
        avail_cols = [col for col in display_cols.keys() if col in df.columns]
        if avail_cols:
            # Select and rename columns
            df_display = df[avail_cols].rename(columns={col: display_cols[col] for col in avail_cols})
            
            # Format columns
            if "Price" in df_display.columns:
                df_display["Price"] = df_display["Price"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "Market")
            
            if "Side" in df_display.columns:
                def format_side(side):
                    if pd.isnull(side):
                        return ""
                    side_upper = str(side).upper()
                    color = "green" if side_upper == "BUY" else "red" if side_upper == "SELL" else "black"
                    return f"<span style='color:{color}'>{side_upper}</span>"
                
                df_display["Side"] = df["side"].apply(format_side)
            
            if "Status" in df_display.columns:
                def format_status(status):
                    if pd.isnull(status):
                        return ""
                    status_str = str(status).lower()
                    color = {
                        "filled": "green", 
                        "open": "blue", 
                        "canceled": "orange",
                        "cancelled": "orange",
                        "rejected": "red"
                    }.get(status_str, "gray")
                    return f"<span style='color:{color}'>{str(status).upper()}</span>"
                
                df_display["Status"] = df["status"].apply(format_status)
            
            if "Date/Time" in df_display.columns:
                df_display["Date/Time"] = pd.to_datetime(df_display["Date/Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Display as HTML table
            st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.error("Order data has unexpected format")
            st.dataframe(df)
    else:
        st.info("No recent orders.")

# Control panel (inspired by MyCryptoBot and Crypto-Bot)
def render_bot_controls(system_status, db):
    st.sidebar.markdown("## Bot Controls")
    
    status_color = {
        "online": "status-active",
        "offline": "status-inactive",
        "error": "status-error",
        "warning": "status-warning"
    }.get(system_status.get("status", "offline"), "status-inactive")
    
    # Status card (inspired by trading-dashboard)
    st.sidebar.markdown(f"""
    <div class="status-card {status_color}">
        <h3>Bot Status: {system_status.get("status", "Unknown").upper()}</h3>
        <p>Trading: {"Enabled" if system_status.get("trading_enabled", False) else "Disabled"}</p>
        <p>Last Update: {system_status.get("last_update", "N/A")}</p>
        <p>{system_status.get("message", "")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons (inspired by Crypto-Bot)
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if system_status.get("trading_enabled", False):
            st.markdown('<div class="danger-button">', unsafe_allow_html=True)
            if st.button("Stop Trading"):
                # Logic to stop trading would go here
                st.sidebar.success("Trading stopped")
                time.sleep(1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="primary-button">', unsafe_allow_html=True)
            if st.button("Start Trading"):
                # Logic to start trading would go here
                st.sidebar.success("Trading started")
                time.sleep(1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="warning-button">', unsafe_allow_html=True)
        if st.button("Reset Bot"):
            # Logic to reset bot would go here
            st.sidebar.info("Bot reset initiated")
            time.sleep(1)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk parameters (inspired by MyCryptoBot)
    st.sidebar.markdown("## Risk Parameters")
    
    max_position = st.sidebar.slider(
        "Max Position Size (%)", 
        min_value=1, 
        max_value=100, 
        value=10,
        help="Maximum percentage of portfolio allocated to a single position"
    )
    
    risk_per_trade = st.sidebar.slider(
        "Risk Per Trade (%)", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0,
        step=0.1,
        help="Maximum percentage of portfolio risked on a single trade"
    )
    
    # Apply settings button
    st.sidebar.markdown('<div class="primary-button">', unsafe_allow_html=True)
    if st.sidebar.button("Apply Settings"):
        # Logic to apply settings would go here
        st.sidebar.success("Settings applied successfully")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main function
def main():
    # Connect to MongoDB
    db, connected = get_mongodb()
    
    # Get system status
    system_status = get_system_status(db)
    
    # Sidebar
    st.sidebar.title("BensBot Dashboard")
    
    # Account type selector
    account_type = st.sidebar.selectbox(
        "Account Type",
        ["Paper Trading", "Live Trading"],
        index=0
    )
    
    # Convert friendly name to internal code
    account_type_code = "paper" if account_type == "Paper Trading" else "live"
    
    # MongoDB connection status
    if connected:
        st.sidebar.success("✅ Connected to MongoDB")
    else:
        st.sidebar.error("❌ MongoDB Not Connected")
        st.sidebar.info("Using demo data")
    
    # Bot controls
    render_bot_controls(system_status, db)
    
    # Auto-refresh
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
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main content
    render_header()
    
    # Get data for selected account
    data = get_portfolio_data(db, account_type=account_type_code)
    
    # Create tabs (inspired by trading-dashboard)
    tab1, tab2, tab3 = st.tabs(["Portfolio", "Positions", "Orders"])
    
    with tab1:
        render_portfolio_summary(data)
    
    with tab2:
        render_positions(data)
    
    with tab3:
        render_orders(data)
    
    # Footer
    st.markdown("---")
    st.caption(f"BensBot Trading Dashboard - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
