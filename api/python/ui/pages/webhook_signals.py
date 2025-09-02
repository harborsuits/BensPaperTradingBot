#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Signals Dashboard Page

Displays and manages external trading signals from various sources:
- TradingView webhooks
- Alpaca API alerts
- Finnhub streaming data
- Other webhook integrations
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.ui.panels.webhook_panel import render_webhook_panel
from trading_bot.strategies.external_signal_strategy import ExternalSignalStrategy, SignalSource

logger = logging.getLogger(__name__)

def render_external_apis_section():
    """Render the external APIs configuration and signal integration section."""
    st.header("External API Integrations")
    
    # Create tabs for different API integrations
    alpaca_tab, finnhub_tab = st.tabs(["Alpaca", "Finnhub"])
    
    # Alpaca Tab
    with alpaca_tab:
        st.subheader("Alpaca Integration")
        
        # Check if external signal strategy exists
        external_signal_strategy = None
        try:
            external_signal_strategy = ServiceRegistry.get_service('external_signal_strategy')
        except Exception as e:
            logger.error(f"Error getting external signal strategy: {str(e)}")
        
        if not external_signal_strategy:
            st.warning("External signal strategy not found. Initialize it in the Webhook Panel first.")
            return
        
        # Alpaca configuration
        st.markdown("### Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            alpaca_enabled = st.checkbox("Enable Alpaca Signals", 
                                       value=external_signal_strategy.get_source_enabled(SignalSource.ALPACA))
            
            # Input fields
            alpaca_api_key = st.text_input("Alpaca API Key", 
                                         type="password", 
                                         value=external_signal_strategy.get_source_config(SignalSource.ALPACA).get("api_key", ""))
            
            alpaca_secret_key = st.text_input("Alpaca Secret Key", 
                                            type="password",
                                            value=external_signal_strategy.get_source_config(SignalSource.ALPACA).get("secret_key", ""))
        
        with col2:
            alpaca_paper = st.checkbox("Paper Trading", 
                                     value=external_signal_strategy.get_source_config(SignalSource.ALPACA).get("paper", True))
            
            trade_signals = st.checkbox("Auto-Trade Alpaca Signals", 
                                       value=external_signal_strategy.get_source_config(SignalSource.ALPACA).get("auto_trade", False))
            
            signal_types = st.multiselect("Signal Types to Process",
                                         options=["trade_updates", "account_updates", "price_alerts"],
                                         default=external_signal_strategy.get_source_config(SignalSource.ALPACA).get("signal_types", ["trade_updates"]))
        
        # Save button
        if st.button("Save Alpaca Configuration"):
            try:
                # Update source config
                alpaca_config = {
                    "enabled": alpaca_enabled,
                    "api_key": alpaca_api_key,
                    "secret_key": alpaca_secret_key,
                    "paper": alpaca_paper,
                    "auto_trade": trade_signals,
                    "signal_types": signal_types
                }
                
                external_signal_strategy.update_source_config(SignalSource.ALPACA, alpaca_config)
                
                if alpaca_enabled:
                    # If we're enabling, try to connect
                    if external_signal_strategy.initialize_source(SignalSource.ALPACA):
                        st.success("Alpaca configuration saved and connected successfully!")
                    else:
                        st.error("Failed to connect to Alpaca. Check your API credentials.")
                else:
                    st.success("Alpaca configuration saved. Service is disabled.")
                    
            except Exception as e:
                st.error(f"Error saving Alpaca configuration: {str(e)}")
        
        # Display recent Alpaca signals if available
        st.markdown("### Recent Alpaca Signals")
        signals = external_signal_strategy.get_signals(source=SignalSource.ALPACA, limit=10)
        
        if not signals:
            st.info("No Alpaca signals received yet.")
        else:
            # Convert signals to dataframe
            signals_data = []
            for signal in signals:
                signals_data.append({
                    "Timestamp": signal.timestamp,
                    "Symbol": signal.symbol,
                    "Type": signal.signal_type.value,
                    "Direction": signal.direction.value,
                    "Price": signal.price,
                    "Processed": "✅" if signal.processed else "❌"
                })
            
            df = pd.DataFrame(signals_data)
            
            # Format timestamp
            df["Timestamp"] = df["Timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Finnhub Tab
    with finnhub_tab:
        st.subheader("Finnhub Integration")
        
        if not external_signal_strategy:
            st.warning("External signal strategy not found. Initialize it in the Webhook Panel first.")
            return
        
        # Finnhub configuration
        st.markdown("### Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            finnhub_enabled = st.checkbox("Enable Finnhub Signals", 
                                       value=external_signal_strategy.get_source_enabled(SignalSource.FINNHUB))
            
            finnhub_api_key = st.text_input("Finnhub API Key", 
                                          type="password",
                                          value=external_signal_strategy.get_source_config(SignalSource.FINNHUB).get("api_key", ""))
        
        with col2:
            trade_signals = st.checkbox("Auto-Trade Finnhub Signals", 
                                       value=external_signal_strategy.get_source_config(SignalSource.FINNHUB).get("auto_trade", False))
            
            # Symbols to monitor
            symbols = st.text_input("Symbols to Monitor (comma-separated)",
                                   value=",".join(external_signal_strategy.get_source_config(SignalSource.FINNHUB).get("symbols", ["AAPL", "MSFT", "AMZN"])))
            
            # Signal types
            signal_types = st.multiselect("Signal Types to Process",
                                         options=["trade", "news", "earnings", "price_alert"],
                                         default=external_signal_strategy.get_source_config(SignalSource.FINNHUB).get("signal_types", ["trade"]))
        
        # Save button
        if st.button("Save Finnhub Configuration"):
            try:
                # Update source config
                finnhub_config = {
                    "enabled": finnhub_enabled,
                    "api_key": finnhub_api_key,
                    "auto_trade": trade_signals,
                    "symbols": [s.strip() for s in symbols.split(",") if s.strip()],
                    "signal_types": signal_types
                }
                
                external_signal_strategy.update_source_config(SignalSource.FINNHUB, finnhub_config)
                
                if finnhub_enabled:
                    # If we're enabling, try to connect
                    if external_signal_strategy.initialize_source(SignalSource.FINNHUB):
                        st.success("Finnhub configuration saved and connected successfully!")
                    else:
                        st.error("Failed to connect to Finnhub. Check your API key.")
                else:
                    st.success("Finnhub configuration saved. Service is disabled.")
                    
            except Exception as e:
                st.error(f"Error saving Finnhub configuration: {str(e)}")
        
        # Display recent Finnhub signals if available
        st.markdown("### Recent Finnhub Signals")
        signals = external_signal_strategy.get_signals(source=SignalSource.FINNHUB, limit=10)
        
        if not signals:
            st.info("No Finnhub signals received yet.")
        else:
            # Convert signals to dataframe
            signals_data = []
            for signal in signals:
                signals_data.append({
                    "Timestamp": signal.timestamp,
                    "Symbol": signal.symbol,
                    "Type": signal.signal_type.value,
                    "Direction": signal.direction.value,
                    "Price": signal.price,
                    "Processed": "✅" if signal.processed else "❌"
                })
            
            df = pd.DataFrame(signals_data)
            
            # Format timestamp
            df["Timestamp"] = df["Timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            
            st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Main function for the webhook signals page."""
    st.set_page_config(
        page_title="External Signals & Webhooks",
        page_icon="📶",
        layout="wide"
    )
    
    st.title("External Signals & Webhooks")
    st.markdown("""
    This page allows you to manage external trading signals from various sources:
    - TradingView Webhooks
    - Alpaca API
    - Finnhub Data Feed
    - Custom Signal Sources
    """)
    
    # Create tabs for Webhook Panel and External APIs
    tab1, tab2 = st.tabs(["Webhook Signals", "API Integrations"])
    
    with tab1:
        render_webhook_panel()
    
    with tab2:
        render_external_apis_section()


if __name__ == "__main__":
    main()
