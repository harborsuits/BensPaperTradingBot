#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Manual Controls Component

Provides controls for manually managing the trading system.
"""

import streamlit as st
import time
import os
import logging

logger = logging.getLogger("BensBot-ManualControls")

def render_manual_controls():
    """Render the manual controls component"""
    
    # Get data service for control actions
    data_service = st.session_state.data_service
    
    # Trading control section
    st.subheader("Trading Controls")
    
    # Start/Pause/Stop trading buttons
    if not st.session_state.trading_active:
        # Trading is paused, show Start button
        if st.button("▶️ Start Trading"):
            with st.spinner("Starting trading system..."):
                # Call data service to start trading
                success, error = data_service.start_trading()
                
                if success:
                    st.session_state.trading_active = True
                    st.success("Trading system started successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to start trading: {error if error else 'Unknown error'}")
    else:
        # Trading is active, show Pause and Stop buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏸️ Pause Trading"):
                with st.spinner("Pausing trading system..."):
                    # Call data service to pause trading
                    success, error = data_service.pause_trading()
                    
                    if success:
                        st.session_state.trading_active = False
                        st.success("Trading system paused successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Failed to pause trading: {error if error else 'Unknown error'}")
        
        with col2:
            if st.button("⏹️ Stop Trading"):
                with st.spinner("Stopping trading system..."):
                    # Call data service to stop trading
                    success, error = data_service.stop_trading()
                    
                    if success:
                        st.session_state.trading_active = False
                        st.success("Trading system stopped successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Failed to stop trading: {error if error else 'Unknown error'}")
    
    # Emergency and maintenance controls
    st.subheader("Emergency Controls")
    
    # Close all positions button (emergency)
    close_positions = st.button("❗ Close All Positions", help="Close all open positions immediately (emergency action)")
    
    if close_positions:
        # Ask for confirmation
        confirm = st.checkbox("I confirm I want to close all positions immediately", key="confirm_close")
        
        if confirm:
            with st.spinner("Closing all positions..."):
                # Call data service to close all positions
                success, error = data_service.close_all_positions()
                
                if success:
                    st.success("All positions closed successfully!")
                    # Update session state to reflect empty positions
                    st.session_state.portfolio['positions'] = []
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to close positions: {error if error else 'Unknown error'}")
    
    # Restart trading loop button
    restart = st.button("🔄 Restart Trading Loop", help="Restart the main trading loop (for maintenance)")
    
    if restart:
        with st.spinner("Restarting trading loop..."):
            # Call data service to restart trading loop
            success, error = data_service.restart_trading_loop()
            
            if success:
                st.success("Trading loop restarted successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Failed to restart trading loop: {error if error else 'Unknown error'}")
    
    # Upload new strategy section
    st.subheader("Upload Strategy")
    
    # Strategy file uploader
    uploaded_file = st.file_uploader("Upload New Strategy", type=["py", "json"], help="Upload a new strategy file")
    
    if uploaded_file is not None:
        with st.spinner("Processing strategy file..."):
            # Call data service to upload strategy
            success, error = data_service.upload_strategy(uploaded_file)
            
            if success:
                st.success(f"Strategy '{uploaded_file.name}' uploaded successfully!")
                # Clear the uploader
                uploaded_file = None
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Failed to upload strategy: {error if error else 'Unknown error'}")
    
    # System status
    st.subheader("System Status")
    
    # Display current trading status
    status_color = "green" if st.session_state.trading_active else "orange"
    status_text = "Active" if st.session_state.trading_active else "Paused"
    
    st.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div style='width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 8px;'></div>"
        f"<div>Trading Status: <strong>{status_text}</strong></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Display broker connection status
    connection_status, connection_error = data_service.get_connection_status()
    connection_color = "green" if connection_status == "Connected" else "red"
    
    st.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div style='width: 12px; height: 12px; border-radius: 50%; background-color: {connection_color}; margin-right: 8px;'></div>"
        f"<div>Broker Connection: <strong>{connection_status}</strong></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Display last refresh time
    if st.session_state.last_update_time:
        last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_update_time))
        st.markdown(f"Last Update: {last_update}")
    
    # Force refresh button
    if st.button("🔄 Refresh Data"):
        with st.spinner("Refreshing data..."):
            data_service.refresh_data()
            st.session_state.last_update_time = time.time()
            st.success("Data refreshed!")
            time.sleep(0.5)
            st.rerun()
