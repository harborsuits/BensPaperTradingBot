#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webhook Panel - Streamlit UI component for displaying and managing external signals.

This panel shows incoming webhook signals from TradingView and other external sources,
allows configuration of webhook settings, and provides a history of received signals.
"""

import json
import logging
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.strategies.external_signal_strategy import (
    ExternalSignalStrategy, ExternalSignal, SignalSource, SignalType, Direction
)

logger = logging.getLogger(__name__)


def format_timestamp(timestamp: datetime) -> str:
    """Format a timestamp in a readable way."""
    now = datetime.now()
    delta = now - timestamp
    
    if delta < timedelta(minutes=1):
        return "Just now"
    elif delta < timedelta(hours=1):
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} min{'s' if minutes != 1 else ''} ago"
    elif delta < timedelta(days=1):
        hours = int(delta.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        return timestamp.strftime("%Y-%m-%d %H:%M")


def render_webhook_panel():
    """
    Render the webhook panel in the Streamlit UI.
    """
    st.header("External Signals & Webhooks")
    
    # Try to get the external signal strategy from the service registry
    external_signal_strategy = None
    try:
        external_signal_strategy = ServiceRegistry.get_service('external_signal_strategy')
        if not isinstance(external_signal_strategy, ExternalSignalStrategy):
            external_signal_strategy = None
    except Exception as e:
        logger.error(f"Error getting external signal strategy: {str(e)}")
    
    if not external_signal_strategy:
        st.warning("External signal strategy not found in the service registry.")
        if st.button("Initialize External Signal Strategy"):
            try:
                external_signal_strategy = ExternalSignalStrategy(register_webhook=True)
                ServiceRegistry.register('external_signal_strategy', external_signal_strategy)
                st.success("External signal strategy initialized and registered.")
                st.rerun()
            except Exception as e:
                st.error(f"Error initializing external signal strategy: {str(e)}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Signal History", "Statistics", "Configuration"])
    
    # TAB 1: Signal History
    with tab1:
        # Filters for signals
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique symbols from signals
            symbols = list(set(signal.symbol for signal in external_signal_strategy.signals))
            symbols.sort()
            symbols = ["All Symbols"] + symbols
            selected_symbol = st.selectbox("Symbol", symbols)
        
        with col2:
            # Source filter
            sources = [source.value for source in SignalSource]
            sources = ["All Sources"] + sources
            selected_source = st.selectbox("Source", sources)
        
        with col3:
            # Signal type filter
            signal_types = [signal_type.value for signal_type in SignalType]
            signal_types = ["All Types"] + signal_types
            selected_type = st.selectbox("Signal Type", signal_types)
        
        # Get signals with the selected filters
        symbol_filter = None if selected_symbol == "All Symbols" else selected_symbol
        source_filter = None if selected_source == "All Sources" else SignalSource(selected_source)
        type_filter = None if selected_type == "All Types" else SignalType(selected_type)
        
        signals = external_signal_strategy.get_signals(
            symbol=symbol_filter, 
            source=source_filter,
            signal_type=type_filter,
            limit=100
        )
        
        # Display signals as a table
        if not signals:
            st.info("No signals matching the selected filters.")
        else:
            # Convert signals to a dataframe for display
            signal_data = []
            for signal in signals:
                signal_data.append({
                    "Timestamp": signal.timestamp,
                    "Symbol": signal.symbol,
                    "Type": signal.signal_type.value,
                    "Direction": signal.direction.value,
                    "Source": signal.source.value,
                    "Price": signal.price,
                    "Processed": "✅" if signal.processed else "❌",
                    "ID": id(signal)  # For unique identification
                })
            
            df = pd.DataFrame(signal_data)
            df["Timestamp"] = df["Timestamp"].apply(format_timestamp)
            
            # Display the table
            st.dataframe(
                df[["Timestamp", "Symbol", "Type", "Direction", "Source", "Price", "Processed"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Allow viewing details of a signal
            if signal_data:
                selected_signal_id = st.selectbox(
                    "Select a signal to view details:",
                    options=[s["ID"] for s in signal_data],
                    format_func=lambda x: f"{signal_data[next(i for i, s in enumerate(signal_data) if s['ID'] == x)]['Symbol']} - {signal_data[next(i for i, s in enumerate(signal_data) if s['ID'] == x)]['Type']} ({signal_data[next(i for i, s in enumerate(signal_data) if s['ID'] == x)]['Timestamp']})"
                )
                
                selected_signal = next((s for s in signals if id(s) == selected_signal_id), None)
                if selected_signal:
                    with st.expander("Signal Details", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Symbol:** {selected_signal.symbol}")
                            st.markdown(f"**Type:** {selected_signal.signal_type.value}")
                            st.markdown(f"**Direction:** {selected_signal.direction.value}")
                            st.markdown(f"**Source:** {selected_signal.source.value}")
                            
                        with col2:
                            st.markdown(f"**Timestamp:** {selected_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            st.markdown(f"**Price:** {selected_signal.price or 'N/A'}")
                            st.markdown(f"**Processed:** {'Yes' if selected_signal.processed else 'No'}")
                            st.markdown(f"**Result:** {selected_signal.result or 'N/A'}")
                        
                        # Display metadata and raw payload
                        if selected_signal.metadata:
                            st.subheader("Metadata")
                            st.json(selected_signal.metadata)
                        
                        if selected_signal.raw_payload:
                            st.subheader("Raw Payload")
                            st.json(selected_signal.raw_payload)
    
    # TAB 2: Statistics
    with tab2:
        signal_stats = external_signal_strategy.get_signal_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Signals", signal_stats["total_signals"])
            st.metric("Processed Signals", signal_stats["processed_count"])
            
            # Chart by source
            if signal_stats["by_source"]:
                st.subheader("Signals by Source")
                source_df = pd.DataFrame({
                    "Source": list(signal_stats["by_source"].keys()),
                    "Count": list(signal_stats["by_source"].values())
                })
                st.bar_chart(source_df.set_index("Source"))
        
        with col2:
            # Chart by type
            if signal_stats["by_type"]:
                st.subheader("Signals by Type")
                type_df = pd.DataFrame({
                    "Type": list(signal_stats["by_type"].keys()),
                    "Count": list(signal_stats["by_type"].values())
                })
                st.bar_chart(type_df.set_index("Type"))
            
            # Chart by symbol
            if signal_stats["by_symbol"]:
                st.subheader("Signals by Symbol")
                symbol_df = pd.DataFrame({
                    "Symbol": list(signal_stats["by_symbol"].keys()),
                    "Count": list(signal_stats["by_symbol"].values())
                })
                st.bar_chart(symbol_df.set_index("Symbol"))
    
    # TAB 3: Configuration
    with tab3:
        st.subheader("Webhook Settings")
        
        # Get webhook handler from service registry
        webhook_handler = None
        try:
            webhook_handler = ServiceRegistry.get_service('webhook_handler')
        except Exception as e:
            logger.error(f"Error getting webhook handler: {str(e)}")
        
        if webhook_handler:
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Webhook URL: http://localhost:{webhook_handler.port}/{webhook_handler.path}")
                st.code(f"curl -X POST http://localhost:{webhook_handler.port}/{webhook_handler.path} \\\n"
                        f"  -H \"Content-Type: application/json\" \\\n"
                        f"  -d '{{\"symbol\": \"EURUSD\", \"action\": \"buy\", \"price\": 1.1234}}'")
            
            with col2:
                if st.button("Test Webhook"):
                    try:
                        import requests
                        response = requests.post(
                            f"http://localhost:{webhook_handler.port}/{webhook_handler.path}",
                            json={
                                "symbol": "EURUSD",
                                "action": "buy",
                                "price": 1.1234,
                                "timestamp": datetime.now().isoformat(),
                                "source": "test"
                            }
                        )
                        if response.status_code == 200:
                            st.success("Test webhook sent successfully!")
                        else:
                            st.error(f"Error sending test webhook: {response.text}")
                    except Exception as e:
                        st.error(f"Error sending test webhook: {str(e)}")
        else:
            st.warning("Webhook handler not found in the service registry.")
        
        st.subheader("Strategy Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_trade = st.checkbox("Auto-Trade Signals", 
                                    value=external_signal_strategy.auto_trade,
                                    help="Automatically generate trades from received signals")
            
            max_history = st.number_input("Max Signal History", 
                                         min_value=10, 
                                         max_value=1000,
                                         value=external_signal_strategy.max_history)
        
        with col2:
            st.write("Signal Filters")
            # Add UI for configuring signal filters
            filter_symbols = st.text_input("Allowed Symbols (comma-separated)",
                                          value=",".join(external_signal_strategy.signal_filters.get("symbols", [])))
        
        # Apply settings button
        if st.button("Apply Settings"):
            try:
                # Update strategy parameters
                params = {
                    "auto_trade": auto_trade,
                    "max_history": max_history,
                    "signal_filters": {
                        "symbols": [s.strip() for s in filter_symbols.split(",")] if filter_symbols else []
                    }
                }
                external_signal_strategy.update_parameters(params)
                st.success("Settings updated successfully!")
            except Exception as e:
                st.error(f"Error updating settings: {str(e)}")
        
        # Danger zone
        st.subheader("Danger Zone")
        if st.button("Clear Signal History", type="primary"):
            external_signal_strategy.clear_signals()
            st.success("Signal history cleared.")
            st.rerun()


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Webhook Panel Test", layout="wide")
    render_webhook_panel()
