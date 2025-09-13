"""
Webhook Signal Monitor Component

This component displays incoming webhook signals from TradingView and other sources.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

def render_webhook_monitor(data_service, max_signals: int = 20, account_type: str = None):
    """
    Render the webhook signal monitor component.
    
    Args:
        data_service: Data service for fetching webhook data
        max_signals: Maximum number of signals to display
        account_type: Type of account to filter signals by
    """
    # Get webhook signal data filtered by account type
    signals_df = data_service.get_webhook_signals(max_signals=max_signals, account_type=account_type)
    
    if signals_df.empty:
        st.info("No webhook signals to display.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Symbol filter
        if 'symbol' in signals_df.columns:
            symbols = ['All Symbols'] + sorted(signals_df['symbol'].unique().tolist())
            selected_symbol = st.selectbox("Filter by Symbol", symbols)
    
    with col2:
        # Source filter
        if 'source' in signals_df.columns:
            sources = ['All Sources'] + sorted(signals_df['source'].unique().tolist())
            selected_source = st.selectbox("Filter by Source", sources)
    
    with col3:
        # Refresh button
        if st.button("Refresh", key="refresh_signals"):
            st.rerun()
    
    # Apply filters
    filtered_df = signals_df.copy()
    
    if 'symbol' in filtered_df.columns and selected_symbol != 'All Symbols':
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
    
    if 'source' in filtered_df.columns and selected_source != 'All Sources':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    if filtered_df.empty:
        st.info("No signals match the selected filters.")
        return
    
    # Format the DataFrame for display
    display_df = filtered_df.copy()
    
    # Format timestamp
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df['timestamp'] = display_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add status styling
    if 'status' in display_df.columns:
        display_df['status_style'] = display_df['status'].apply(
            lambda x: "success" if x == "Processed" else
                      "warning" if x == "Ignored" else ""
        )
    
    # Add action styling
    if 'action_taken' in display_df.columns:
        display_df['action_style'] = display_df['action_taken'].apply(
            lambda x: "buy" if x == "BUY" else
                      "sell" if x == "SELL" else "none"
        )
    
    # Display the signals in a styled table
    for idx, row in display_df.iterrows():
        timestamp = row.get('timestamp', '')
        source = row.get('source', '')
        symbol = row.get('symbol', '')
        message = row.get('message', '')
        status = row.get('status', '')
        status_style = row.get('status_style', '')
        action = row.get('action_taken', '')
        action_style = row.get('action_style', '')
        
        # Status color
        status_color = "#4CAF50" if status_style == "success" else "#FF9800" if status_style == "warning" else "#9E9E9E"
        
        # Action color
        action_color = "#4CAF50" if action_style == "buy" else "#F44336" if action_style == "sell" else "#9E9E9E"
        
        # Create signal card
        st.markdown(f"""
        <div style="
            background-color: white; 
            border-radius: 5px; 
            padding: 12px; 
            margin-bottom: 8px; 
            border-left: 4px solid {status_color};
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; font-size: 1.1em;">{symbol}</span>
                <span style="color: #757575; font-size: 0.8em;">{timestamp}</span>
            </div>
            <div style="margin-bottom: 8px; font-size: 0.9em;">
                {message}
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85em;">
                <span>Source: <b>{source}</b></span>
                <div>
                    <span style="
                        padding: 2px 6px; 
                        border-radius: 3px; 
                        background-color: rgba({status_color.lstrip('#')[:2] if len(status_color) > 4 else status_color.lstrip('#')[0]+status_color.lstrip('#')[0], 16}, 
                                        {status_color.lstrip('#')[2:4] if len(status_color) > 4 else status_color.lstrip('#')[1]+status_color.lstrip('#')[1], 16}, 
                                        {status_color.lstrip('#')[4:6] if len(status_color) > 4 else status_color.lstrip('#')[2]+status_color.lstrip('#')[2], 16}, 0.2);
                        color: {status_color};
                        margin-right: 8px;
                    ">
                        {status}
                    </span>
                    <span style="
                        padding: 2px 6px; 
                        border-radius: 3px; 
                        background-color: rgba({action_color.lstrip('#')[:2] if len(action_color) > 4 else action_color.lstrip('#')[0]+action_color.lstrip('#')[0], 16}, 
                                        {action_color.lstrip('#')[2:4] if len(action_color) > 4 else action_color.lstrip('#')[1]+action_color.lstrip('#')[1], 16}, 
                                        {action_color.lstrip('#')[4:6] if len(action_color) > 4 else action_color.lstrip('#')[2]+action_color.lstrip('#')[2], 16}, 0.2);
                        color: {action_color};
                    ">
                        {action}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Signal summary
    st.subheader("Signal Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Signals by symbol
        if 'symbol' in signals_df.columns:
            symbol_counts = signals_df['symbol'].value_counts().reset_index()
            symbol_counts.columns = ['Symbol', 'Count']
            
            # Create a horizontal bar chart
            st.markdown("### Signals by Symbol")
            
            # Format as a simple bar chart using HTML
            for idx, row in symbol_counts.iterrows():
                # Calculate percentage width (max 95%)
                width_pct = min(95, row['Count'] / symbol_counts['Count'].max() * 100)
                
                st.markdown(f"""
                <div style="margin-bottom: 4px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 80px; text-align: left;">{row['Symbol']}</div>
                        <div style="flex-grow: 1;">
                            <div style="background-color: #2196F3; height: 20px; width: {width_pct}%; border-radius: 2px;"></div>
                        </div>
                        <div style="width: 30px; text-align: right; margin-left: 8px;">{row['Count']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Action distribution
        if 'action_taken' in signals_df.columns:
            action_counts = signals_df['action_taken'].value_counts().reset_index()
            action_counts.columns = ['Action', 'Count']
            
            # Create a horizontal bar chart
            st.markdown("### Actions Taken")
            
            # Format as a simple bar chart using HTML with color coding
            for idx, row in action_counts.iterrows():
                # Calculate percentage width (max 95%)
                width_pct = min(95, row['Count'] / action_counts['Count'].max() * 100)
                
                # Determine color based on action
                bar_color = "#4CAF50" if row['Action'] == "BUY" else "#F44336" if row['Action'] == "SELL" else "#9E9E9E"
                
                st.markdown(f"""
                <div style="margin-bottom: 4px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 80px; text-align: left;">{row['Action']}</div>
                        <div style="flex-grow: 1;">
                            <div style="background-color: {bar_color}; height: 20px; width: {width_pct}%; border-radius: 2px;"></div>
                        </div>
                        <div style="width: 30px; text-align: right; margin-left: 8px;">{row['Count']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
