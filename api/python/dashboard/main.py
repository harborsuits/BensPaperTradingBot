"""
BensBot Trading Dashboard

Main entry point for the trading system dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from trading_bot.core.constants import TradingMode
from trading_bot.risk.emergency_controls import EmergencyControls
from trading_bot.data.data_validator import DataValidator


def get_trading_mode():
    """Get the current trading mode from environment."""
    mode_str = os.environ.get("TRADING_MODE", "backtest").upper()
    try:
        return TradingMode[mode_str]
    except KeyError:
        return TradingMode.BACKTEST


def render_system_summary():
    """Render a summary of the trading system status."""
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Example data - would be retrieved from actual system in production
    trading_mode = get_trading_mode()
    account_value = 102350.75
    daily_pnl = 1250.50
    daily_pnl_pct = daily_pnl / (account_value - daily_pnl) * 100
    position_count = 8
    
    with col1:
        st.metric("Trading Mode", trading_mode.name)
        
        if trading_mode == TradingMode.PAPER:
            st.info("🧪 Paper Trading")
        elif trading_mode == TradingMode.LIVE:
            st.error("💵 Live Trading")
        elif trading_mode == TradingMode.STOPPED:
            st.warning("⛔ Trading Stopped")
        else:
            st.success("📊 Backtest Mode")
    
    with col2:
        st.metric("Account Value", f"${account_value:,.2f}")
    
    with col3:
        st.metric("Daily P&L", f"${daily_pnl:,.2f}", f"{daily_pnl_pct:.2f}%")
    
    with col4:
        st.metric("Active Positions", position_count)


def render_quick_actions():
    """Render quick action buttons for common operations."""
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🛑 Emergency Stop"):
            st.error("This would activate the emergency kill switch")
            # In a real implementation: EmergencyControls().activate_kill_switch("Manual activation from dashboard")
    
    with col2:
        if st.button("📊 Generate Performance Report"):
            st.info("This would generate a performance report")
            # In a real implementation: call reporting service
    
    with col3:
        if st.button("🔄 Refresh Market Data"):
            st.info("This would refresh market data")
            # In a real implementation: trigger market data refresh


def render_recent_activity():
    """Render recent activity feed."""
    st.subheader("Recent Activity")
    
    # Sample activity data - would be real system events in production
    activities = [
        {"time": "23:30:15", "type": "TRADE", "description": "Sold 50 MSFT @ $287.65"},
        {"time": "23:25:42", "type": "ALERT", "description": "Data quality warning for TSLA: Stale data detected"},
        {"time": "23:20:18", "type": "SYSTEM", "description": "Market Data Source reconnected successfully"},
        {"time": "23:15:30", "type": "TRADE", "description": "Bought 100 AAPL @ $172.35"},
        {"time": "23:10:55", "type": "STRATEGY", "description": "Mean Reversion strategy activated for NVDA"},
        {"time": "23:05:22", "type": "ALERT", "description": "Position limit warning: AMZN approaching max allocation"}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(activities)
    
    # Apply styling based on activity type
    def style_activity(df):
        styles = []
        for i, row in df.iterrows():
            style = {}
            
            # Style based on type
            if row['type'] == 'TRADE':
                style['type'] = 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif row['type'] == 'ALERT':
                style['type'] = 'background-color: #fff3cd; color: #856404; font-weight: bold'
            elif row['type'] == 'SYSTEM':
                style['type'] = 'background-color: #d1ecf1; color: #0c5460; font-weight: bold'
            elif row['type'] == 'STRATEGY':
                style['type'] = 'background-color: #e2e3e5; color: #383d41; font-weight: bold'
            
            styles.append(style)
        
        return pd.DataFrame(styles, index=df.index)
    
    # Display the styled table
    st.dataframe(df.style.apply(style_activity, axis=1), use_container_width=True)


def render_performance_snapshot():
    """Render a snapshot of recent performance."""
    st.subheader("Performance Snapshot")
    
    # Sample performance data - would be real metrics in production
    dates = pd.date_range(start=datetime.now() - timedelta(days=10), end=datetime.now(), freq='D')
    
    # Generate some realistic daily P&L values with a slight positive bias
    np.random.seed(42)  # For reproducible results
    daily_pnl = np.random.normal(200, 1000, len(dates))  # Mean $200, std $1000
    
    # Calculate cumulative P&L
    cumulative_pnl = np.cumsum(daily_pnl)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Daily P&L': daily_pnl,
        'Cumulative P&L': cumulative_pnl
    })
    
    # Create chart
    fig = go.Figure()
    
    # Add line for cumulative P&L
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative P&L'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='green', width=3)
    ))
    
    # Add bars for daily P&L
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Daily P&L'],
        name='Daily P&L',
        marker_color=['green' if p >= 0 else 'red' for p in df['Daily P&L']]
    ))
    
    # Set y-axis to start at the minimum cumulative value
    y_min = min(0, df['Cumulative P&L'].min() * 1.1)
    y_max = max(0, df['Cumulative P&L'].max() * 1.1)
    
    fig.update_layout(
        title="10-Day Performance",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[y_min, y_max])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_alert_summary():
    """Render a summary of active alerts."""
    st.subheader("Active Alerts")
    
    # Sample alert data - would be real alerts in production
    alerts = [
        {"priority": "HIGH", "type": "DATA_QUALITY", "message": "Stale market data detected for TSLA", "time": "5 min ago"},
        {"priority": "MEDIUM", "type": "POSITION_LIMIT", "message": "AMZN position approaching maximum allocation", "time": "15 min ago"},
        {"priority": "LOW", "type": "SYSTEM", "message": "Elevated API latency with Alpaca", "time": "30 min ago"}
    ]
    
    if alerts:
        for alert in alerts:
            if alert["priority"] == "HIGH":
                st.error(f"**{alert['type']}**: {alert['message']} ({alert['time']})")
            elif alert["priority"] == "MEDIUM":
                st.warning(f"**{alert['type']}**: {alert['message']} ({alert['time']})")
            else:
                st.info(f"**{alert['type']}**: {alert['message']} ({alert['time']})")
    else:
        st.success("✅ No active alerts")


def main():
    """Main entry point for the dashboard."""
    st.set_page_config(
        page_title="Trading Bot Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("BensBot Trading Dashboard")
    st.write("Comprehensive trading system monitoring and control")
    
    # Display date and time in sidebar
    st.sidebar.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Navigation
    st.sidebar.title("Navigation")
    st.sidebar.info(
        "Select a dashboard page from the dropdown in the sidebar to view detailed information "
        "about specific aspects of the trading system."
    )
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.experimental_rerun()
    
    # System status summary
    render_system_summary()
    
    # Quick action buttons
    render_quick_actions()
    
    # Create two columns for the bottom sections
    col1, col2 = st.columns(2)
    
    with col1:
        render_recent_activity()
    
    with col2:
        render_alert_summary()
    
    # Performance snapshot
    render_performance_snapshot()
    
    # Footer
    st.markdown("---")
    st.caption(
        "**BensBot Trading System** | "
        "Data may be delayed | "
        "For informational purposes only | "
        "Not financial advice"
    )


if __name__ == "__main__":
    main()
