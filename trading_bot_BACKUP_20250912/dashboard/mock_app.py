"""
Mock Streamlit Dashboard Application for Trading Bot

This version uses mock data and doesn't rely on external imports
that might be causing issues with the full trading_bot package.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import os

# Page configuration
st.set_page_config(
    page_title="BensBot Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f9f9f9;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
    }
    .warning {
        color: #FF9800;
        font-weight: bold;
    }
    .paper-mode {
        color: #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .live-mode {
        color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .emergency-button {
        background-color: #F44336;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Mock data generators
def generate_portfolio_summary():
    return {
        "total_equity": random.uniform(95000, 105000),
        "daily_pnl": random.uniform(-2000, 2000),
        "daily_pnl_pct": random.uniform(-2, 2),
        "total_pnl": random.uniform(5000, 15000),
        "total_pnl_pct": random.uniform(5, 15),
        "open_positions_count": random.randint(3, 8)
    }

def generate_system_status():
    return {
        "trading_enabled": random.choice([True, True, True, False]),  # More likely to be enabled
        "uptime": f"{random.randint(1, 24)}h {random.randint(1, 59)}m",
        "cpu_usage": random.uniform(10, 60),
        "memory_usage": random.uniform(20, 70),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_market_context():
    regimes = ["Bullish", "Neutral", "Bearish"]
    weights = [0.4, 0.4, 0.2]  # More likely to be bullish or neutral
    return {
        "market_regime": random.choices(regimes, weights=weights)[0],
        "vix": random.uniform(12, 25),
        "spy_change": random.uniform(-1, 1),
        "sector_performance": {
            "Technology": random.uniform(-1.5, 1.5),
            "Financial": random.uniform(-1.5, 1.5),
            "Healthcare": random.uniform(-1.5, 1.5),
            "Consumer": random.uniform(-1.5, 1.5),
            "Energy": random.uniform(-1.5, 1.5)
        }
    }

def generate_strategies():
    strategies = []
    for i in range(1, 6):
        is_paper = random.choice([True, False])
        strategies.append({
            "id": f"strat_{i}",
            "name": f"Strategy {i}",
            "type": random.choice(["Momentum", "Mean Reversion", "Pattern", "Breakout"]),
            "status": random.choice(["Active", "Paused", "Error"]),
            "mode": "Paper" if is_paper else "Live",
            "pnl_today": random.uniform(-500, 500),
            "pnl_total": random.uniform(-2000, 5000),
            "win_rate": random.uniform(45, 65),
            "trades_today": random.randint(0, 10)
        })
    return strategies

def generate_trade_log(count=20):
    trades = []
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "NVDA", "META", "NFLX"]
    
    for i in range(count):
        direction = random.choice(["BUY", "SELL"])
        entry_price = random.uniform(100, 1000)
        exit_price = entry_price * (1 + random.uniform(-0.05, 0.05))
        pnl = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
        
        # Older trades are more likely to be closed
        is_open = random.random() > (i / count * 0.8)
        
        trades.append({
            "id": f"trade_{i}",
            "timestamp": (datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": random.choice(symbols),
            "direction": direction,
            "size": random.randint(1, 10),
            "entry_price": entry_price,
            "exit_price": exit_price if not is_open else None,
            "pnl": pnl if not is_open else None,
            "status": "Open" if is_open else "Closed",
            "strategy": f"Strategy {random.randint(1, 5)}"
        })
    
    return trades

def generate_broker_balances():
    return {
        "Alpaca": {
            "equity": random.uniform(40000, 60000),
            "cash": random.uniform(20000, 40000),
            "positions_value": random.uniform(20000, 40000),
            "positions_count": random.randint(2, 5)
        },
        "Interactive Brokers": {
            "equity": random.uniform(30000, 50000),
            "cash": random.uniform(10000, 30000),
            "positions_value": random.uniform(10000, 30000),
            "positions_count": random.randint(1, 4)
        }
    }

def generate_alerts(count=5):
    alert_types = ["system", "risk", "strategy", "market"]
    alert_levels = ["info", "warning", "error"]
    alert_messages = [
        "Strategy exceeding max drawdown limit",
        "System CPU usage high",
        "API connection interrupted",
        "Unusual market volatility detected",
        "New trade signal detected",
        "Position size limit reached",
        "Strategy performance below threshold",
        "Data feed delay detected",
        "Broker connection issue"
    ]
    
    alerts = []
    for i in range(count):
        level = random.choice(alert_levels)
        alerts.append({
            "id": f"alert_{i}",
            "timestamp": (datetime.now() - timedelta(minutes=i*random.randint(5, 20))).strftime("%Y-%m-%d %H:%M:%S"),
            "type": random.choice(alert_types),
            "level": level,
            "message": random.choice(alert_messages),
            "acknowledged": random.choice([True, False])
        })
    
    return alerts

def generate_performance_data():
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
    
    # Create cumulative equity curves for each strategy
    base_equity = 10000
    strategies_equity = {}
    strategy_names = [f"Strategy {i}" for i in range(1, 6)]
    
    for strategy in strategy_names:
        daily_returns = [random.uniform(-0.02, 0.03) for _ in range(len(dates))]
        equity_curve = [base_equity]
        
        for ret in daily_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        strategies_equity[strategy] = equity_curve[1:]  # Remove the initial value
    
    # Create a dataframe with the equity curves
    df = pd.DataFrame({"Date": dates})
    for strategy, equity in strategies_equity.items():
        df[strategy] = equity
    
    # Add total portfolio equity
    df["Portfolio"] = df[[col for col in df.columns if col != "Date"]].sum(axis=1)
    
    return df

# Dashboard components
def render_active_strategies(strategies):
    if not strategies:
        st.info("No active strategies")
        return
    
    # Create columns for each strategy attribute
    cols = st.columns(len(strategies))
    
    for i, strategy in enumerate(cols):
        with strategy:
            s = strategies[i]
            mode_class = "paper-mode" if s["mode"] == "Paper" else "live-mode"
            st.markdown(f"### {s['name']}")
            st.markdown(f"**Type:** {s['type']}")
            st.markdown(f"**Status:** {s['status']}")
            st.markdown(f"**Mode:** <span class='{mode_class}'>{s['mode']}</span>", unsafe_allow_html=True)
            
            # Display PnL with color
            pnl_class = "positive" if s["pnl_today"] >= 0 else "negative"
            st.markdown(f"**Today's P&L:** <span class='{pnl_class}'>${s['pnl_today']:.2f}</span>", unsafe_allow_html=True)
            
            # Controls
            col1, col2 = st.columns(2)
            with col1:
                if s["status"] == "Active":
                    st.button(f"Pause {s['name']}", key=f"pause_{i}")
                else:
                    st.button(f"Resume {s['name']}", key=f"resume_{i}")
            with col2:
                st.button(f"Close Positions", key=f"close_{i}")

def render_trade_log(trades, max_trades=None):
    if max_trades:
        trades = trades[:max_trades]
    
    # Convert trades to DataFrame for display
    df = pd.DataFrame(trades)
    if len(df) == 0:
        st.info("No trades to display")
        return
    
    # Format columns
    if 'pnl' in df.columns:
        df['pnl'] = df['pnl'].apply(lambda x: f"${x:.2f}" if x is not None else "")
    
    if 'entry_price' in df.columns:
        df['entry_price'] = df['entry_price'].apply(lambda x: f"${x:.2f}" if x is not None else "")
    
    if 'exit_price' in df.columns:
        df['exit_price'] = df['exit_price'].apply(lambda x: f"${x:.2f}" if x is not None else "")
    
    # Display the dataframe
    st.dataframe(df)

def render_performance_chart(performance_data):
    fig = go.Figure()
    
    # Add a line for each strategy
    for col in performance_data.columns:
        if col != "Date":
            fig.add_trace(go.Scatter(
                x=performance_data["Date"],
                y=performance_data[col],
                mode='lines',
                name=col
            ))
    
    fig.update_layout(
        title="Equity Curves",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        legend_title="Strategy",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_alerts_panel(alerts):
    if not alerts:
        st.info("No alerts to display")
        return
    
    for alert in alerts:
        level_class = {
            "info": "info",
            "warning": "warning",
            "error": "negative"
        }.get(alert["level"], "info")
        
        message = alert["message"]
        timestamp = alert["timestamp"]
        
        st.markdown(f"<div style='padding:10px;margin-bottom:10px;border-left:4px solid {'red' if level_class == 'negative' else 'orange' if level_class == 'warning' else 'blue'};background-color:{'rgba(244, 67, 54, 0.1)' if level_class == 'negative' else 'rgba(255, 152, 0, 0.1)' if level_class == 'warning' else 'rgba(33, 150, 243, 0.1)'}'>"\
                   f"<span class='{level_class}'>{alert['level'].upper()}</span>: {message}<br/>"\
                   f"<small>{timestamp}</small>"\
                   f"</div>", 
                   unsafe_allow_html=True)

def render_manual_override():
    st.button("⚠️ PAUSE ALL TRADING", key="pause_all", help="Immediately pause all trading activity")
    st.button("🔄 RESUME ALL TRADING", key="resume_all", help="Resume all paused trading activity")
    st.button("❌ CLOSE ALL POSITIONS", key="close_all", help="Close all open positions")

def render_broker_balances(broker_balances, simplified=False):
    for broker_name, data in broker_balances.items():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(broker_name)
            st.metric("Total Equity", f"${data['equity']:,.2f}")
            st.metric("Cash", f"${data['cash']:,.2f}")
        
        with col2:
            st.metric("Positions Value", f"${data['positions_value']:,.2f}")
            st.metric("Positions Count", data['positions_count'])
        
        if not simplified:
            # Add a chart showing cash vs positions
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Cash", "Positions"],
                    values=[data['cash'], data['positions_value']],
                    hole=.3
                )
            ])
            
            fig.update_layout(
                title=f"{broker_name} Allocation",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")

# Main app
def main():
    # Generate mock data
    portfolio_summary = generate_portfolio_summary()
    system_status = generate_system_status()
    market_data = generate_market_context()
    strategies = generate_strategies()
    trades = generate_trade_log()
    broker_balances = generate_broker_balances()
    alerts = generate_alerts()
    performance_data = generate_performance_data()
    
    # Sidebar with system status and navigation
    with st.sidebar:
        st.title("BensBot Trading System")
        
        # System Status
        st.subheader("System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if system_status["trading_enabled"]:
                st.success("Trading: Enabled")
            else:
                st.error("Trading: Disabled")
        with col2:
            uptime = system_status.get("uptime", "Unknown")
            st.info(f"Uptime: {uptime}")
        
        # Global stats
        st.subheader("Portfolio Summary")
        
        total_equity = portfolio_summary.get("total_equity", 0)
        daily_pnl = portfolio_summary.get("daily_pnl", 0)
        daily_pnl_pct = daily_pnl / total_equity * 100 if total_equity else 0
        
        st.metric(
            "Total Portfolio Value", 
            f"${total_equity:,.2f}",
            f"{daily_pnl_pct:.2f}% Today"
        )
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Strategy Performance", "Trade Log", "Broker & Positions"]
        )
        
        # Manual Override Panel (Always visible)
        st.subheader("Emergency Controls")
        render_manual_override()
        
        # Connection status - this is a mock version
        st.sidebar.markdown("---")
        st.sidebar.info("⚠️ Using MOCK data - Not connected to live trading system")
    
    # Main content area
    st.title("BensBot Trading Dashboard")
    st.caption("DEMO MODE - Using simulated data")
    
    if page == "Dashboard":
        # Summary Row
        st.subheader("Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total P&L", 
                f"${portfolio_summary.get('total_pnl', 0):,.2f}", 
                f"{portfolio_summary.get('total_pnl_pct', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "Today's P&L", 
                f"${portfolio_summary.get('daily_pnl', 0):,.2f}", 
                f"{portfolio_summary.get('daily_pnl_pct', 0):.2f}%"
            )
        
        with col3:
            st.metric(
                "Open Positions", 
                portfolio_summary.get('open_positions_count', 0)
            )
        
        with col4:
            # Market regime from market context data
            regime = market_data.get('market_regime', 'Neutral')
            regime_colors = {"Bullish": "green", "Bearish": "red", "Neutral": "gray"}
            st.markdown(f"**Market Regime**<br><span style='color:{regime_colors.get(regime, 'gray')};font-size:18px;'>{regime}</span>", unsafe_allow_html=True)
        
        # Main dashboard layout in two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Active Strategies Monitor
            st.subheader("Active Strategies")
            render_active_strategies(strategies)
            
            # Performance Chart
            st.subheader("Performance")
            render_performance_chart(performance_data)
        
        with col2:
            # Alerts Panel
            st.subheader("Recent Alerts")
            render_alerts_panel(alerts)
            
            # Trade Log (Recent only)
            st.subheader("Recent Trades")
            render_trade_log(trades, max_trades=5)
            
            # Broker Balances (Simplified)
            st.subheader("Broker Accounts")
            render_broker_balances(broker_balances, simplified=True)
    
    elif page == "Strategy Performance":
        st.header("Strategy Performance Analysis")
        render_performance_chart(performance_data)
        
        # Strategy metrics
        st.subheader("Strategy Metrics")
        metrics_df = pd.DataFrame({
            "Strategy": [s["name"] for s in strategies],
            "Win Rate": [f"{s['win_rate']:.1f}%" for s in strategies],
            "Total P&L": [f"${s['pnl_total']:.2f}" for s in strategies],
            "Today's P&L": [f"${s['pnl_today']:.2f}" for s in strategies],
            "Trades Today": [s["trades_today"] for s in strategies],
            "Mode": [s["mode"] for s in strategies],
            "Status": [s["status"] for s in strategies]
        })
        
        st.dataframe(metrics_df)
    
    elif page == "Trade Log":
        st.header("Trade Log")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy_filter = st.multiselect(
                "Filter by Strategy",
                options=[s["name"] for s in strategies],
                default=[]
            )
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=["Open", "Closed"],
                default=[]
            )
        with col3:
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=["BUY", "SELL"],
                default=[]
            )
        
        # Apply filtering (in a real app, this would filter the database query)
        filtered_trades = trades
        render_trade_log(filtered_trades)
    
    elif page == "Broker & Positions":
        st.header("Broker Accounts & Positions")
        render_broker_balances(broker_balances)
    
    # Auto-refresh mechanism
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh:
        st.empty()
        time.sleep(5)
        st.experimental_rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption("BensBot Trading System © 2025 - Demo Mode")

if __name__ == "__main__":
    main()
