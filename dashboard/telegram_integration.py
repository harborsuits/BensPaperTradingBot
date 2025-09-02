"""
Telegram Integration Dashboard Module

This module provides UI components for configuring and testing Telegram alerts
from the BensBot dashboard.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Optional, Any

from dashboard.theme import COLORS
from dashboard.components import section_header

# Import the Telegram alert manager if available
try:
    from trading_bot.alerts.telegram_alerts import (
        get_telegram_alert_manager,
        send_risk_alert,
        send_strategy_rotation_alert,
        send_trade_alert,
        send_system_alert
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

def telegram_settings_section():
    """Display and manage Telegram alert settings."""
    section_header("Telegram Alert Configuration", icon="📱")
    
    if not TELEGRAM_AVAILABLE:
        st.error("Telegram integration is not available. Make sure the trading_bot module is properly installed.")
        return
    
    # Get the Telegram alert manager
    telegram_manager = get_telegram_alert_manager()
    
    # Check if Telegram is configured
    is_configured = telegram_manager.is_configured()
    
    # Display current status
    if is_configured:
        st.success("✅ Telegram alerts are configured and ready to use")
    else:
        st.warning("⚠️ Telegram alerts are not configured")
    
    # Create tabs for configuration and testing
    tab1, tab2 = st.tabs(["Configuration", "Test Alerts"])
    
    with tab1:
        # Configuration form
        with st.form("telegram_config_form"):
            st.subheader("Telegram Bot Configuration")
            
            # Get current values if available
            current_token = telegram_manager.bot_token or ""
            current_chat_id = telegram_manager.chat_id or ""
            
            # Input fields
            bot_token = st.text_input(
                "Bot Token", 
                value=current_token,
                help="Enter your Telegram Bot API token obtained from @BotFather",
                type="password"
            )
            
            chat_id = st.text_input(
                "Chat ID",
                value=current_chat_id,
                help="Enter your Telegram chat ID to receive alerts"
            )
            
            st.markdown("""
            **How to set up Telegram alerts:**
            1. Create a new bot through [@BotFather](https://t.me/botfather) on Telegram
            2. Copy the bot token provided by BotFather
            3. Start a chat with your bot
            4. Get your chat ID (you can use [@userinfobot](https://t.me/userinfobot) or [@RawDataBot](https://t.me/RawDataBot))
            5. Enter both values above and save
            """)
            
            # Save button
            submit = st.form_submit_button("Save Configuration")
            
            if submit:
                if bot_token and chat_id:
                    # Save the configuration
                    result = telegram_manager.set_credentials(bot_token, chat_id)
                    if result:
                        st.success("✅ Telegram configuration saved successfully")
                    else:
                        st.error("❌ Failed to save Telegram configuration")
                else:
                    st.error("❌ Both Bot Token and Chat ID are required")
    
    with tab2:
        # Testing section
        st.subheader("Test Telegram Alerts")
        
        if not is_configured:
            st.warning("⚠️ Configure Telegram first before testing alerts")
        else:
            # Connection test
            if st.button("Test Connection"):
                with st.spinner("Testing connection to Telegram..."):
                    result = telegram_manager.test_connection()
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                    else:
                        st.error(f"❌ {result['message']}")
            
            # Alert type selection
            st.subheader("Send Test Alert")
            
            alert_type = st.selectbox(
                "Alert Type",
                options=["Risk Alert", "Strategy Rotation", "Trade Alert", "System Alert"]
            )
            
            # Risk alert form
            if alert_type == "Risk Alert":
                with st.form("risk_alert_test_form"):
                    risk_type = st.selectbox(
                        "Risk Type",
                        options=["Drawdown", "Correlation", "Liquidity", "Volatility", "Position Size", "Sector Exposure"]
                    )
                    
                    risk_level = st.selectbox(
                        "Risk Level",
                        options=["Low", "Medium", "High", "Critical"]
                    )
                    
                    message = st.text_area(
                        "Alert Message",
                        value=f"Test {risk_type.lower()} risk alert from BensBot dashboard"
                    )
                    
                    details = {
                        "Test": "This is a test alert",
                        "Source": "BensBot Dashboard",
                        "Time": "Now"
                    }
                    
                    # Send button
                    submit = st.form_submit_button("Send Test Risk Alert")
                    
                    if submit:
                        with st.spinner("Sending test risk alert..."):
                            result = send_risk_alert(
                                risk_type=risk_type.lower(),
                                risk_level=risk_level.lower(),
                                message=message,
                                details=details
                            )
                            if result["success"]:
                                st.success("✅ Test risk alert sent successfully")
                            else:
                                st.error(f"❌ Failed to send test alert: {result['message']}")
            
            # Strategy rotation alert form
            elif alert_type == "Strategy Rotation":
                with st.form("strategy_rotation_test_form"):
                    trigger = st.selectbox(
                        "Trigger",
                        options=["Market Regime Change", "Risk Threshold", "Performance Decline", "Correlation Alert"]
                    )
                    
                    old_strategies = st.text_input(
                        "Previous Strategies (comma-separated)",
                        value="MomentumStrategy, TrendFollowingStrategy"
                    )
                    
                    new_strategies = st.text_input(
                        "New Strategies (comma-separated)",
                        value="MeanReversionStrategy, LowVolatilityStrategy"
                    )
                    
                    reason = st.text_area(
                        "Rotation Reason",
                        value="Test strategy rotation alert from BensBot dashboard"
                    )
                    
                    # Send button
                    submit = st.form_submit_button("Send Test Rotation Alert")
                    
                    if submit:
                        with st.spinner("Sending test strategy rotation alert..."):
                            result = send_strategy_rotation_alert(
                                trigger=trigger,
                                old_strategies=[s.strip() for s in old_strategies.split(",")],
                                new_strategies=[s.strip() for s in new_strategies.split(",")],
                                reason=reason
                            )
                            if result["success"]:
                                st.success("✅ Test strategy rotation alert sent successfully")
                            else:
                                st.error(f"❌ Failed to send test alert: {result['message']}")
            
            # Trade alert form
            elif alert_type == "Trade Alert":
                with st.form("trade_alert_test_form"):
                    action = st.selectbox(
                        "Action",
                        options=["Buy", "Sell"]
                    )
                    
                    symbol = st.text_input(
                        "Symbol",
                        value="AAPL"
                    )
                    
                    quantity = st.number_input(
                        "Quantity",
                        min_value=1.0,
                        value=100.0,
                        step=1.0
                    )
                    
                    price = st.number_input(
                        "Price",
                        min_value=0.01,
                        value=150.0,
                        step=0.01,
                        format="%.2f"
                    )
                    
                    trade_type = st.selectbox(
                        "Trade Type",
                        options=["Market", "Limit", "Stop", "Stop Limit"]
                    )
                    
                    reason = st.text_area(
                        "Trade Reason",
                        value="Test trade alert from BensBot dashboard"
                    )
                    
                    # Send button
                    submit = st.form_submit_button("Send Test Trade Alert")
                    
                    if submit:
                        with st.spinner("Sending test trade alert..."):
                            result = send_trade_alert(
                                action=action,
                                symbol=symbol,
                                quantity=quantity,
                                price=price,
                                trade_type=trade_type.lower(),
                                reason=reason
                            )
                            if result["success"]:
                                st.success("✅ Test trade alert sent successfully")
                            else:
                                st.error(f"❌ Failed to send test alert: {result['message']}")
            
            # System alert form
            elif alert_type == "System Alert":
                with st.form("system_alert_test_form"):
                    component = st.selectbox(
                        "Component",
                        options=["Strategy Manager", "Risk Manager", "Order Execution", "Data Provider", "Event Bus"]
                    )
                    
                    status = st.selectbox(
                        "Status",
                        options=["Online", "Warning", "Error", "Offline", "Starting"]
                    )
                    
                    severity = st.selectbox(
                        "Severity",
                        options=["Info", "Low", "Medium", "High", "Critical"]
                    )
                    
                    message = st.text_area(
                        "Alert Message",
                        value=f"Test system alert for {component} from BensBot dashboard"
                    )
                    
                    # Send button
                    submit = st.form_submit_button("Send Test System Alert")
                    
                    if submit:
                        with st.spinner("Sending test system alert..."):
                            result = send_system_alert(
                                component=component,
                                status=status.lower(),
                                message=message,
                                severity=severity.lower()
                            )
                            if result["success"]:
                                st.success("✅ Test system alert sent successfully")
                            else:
                                st.error(f"❌ Failed to send test alert: {result['message']}")

def telegram_dashboard():
    """Main Telegram integration dashboard."""
    st.title("Telegram Alert Integration")
    
    telegram_settings_section()

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Telegram Integration - BensBot Dashboard",
        page_icon="📱",
        layout="wide",
    )
    
    telegram_dashboard()
