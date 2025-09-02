#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Trade Log Component

Displays the history of trades executed by the trading bot.
"""

import streamlit as st
import pandas as pd
import datetime

def render_trade_log(limit=None):
    """
    Render the trade log component
    
    Args:
        limit: Optional limit on number of trades to display
    """
    # Get trades from data service
    data_service = st.session_state.data_service
    
    # Add filter controls if no limit is provided (full page view)
    symbol_filter = None
    strategy_filter = None
    date_filter = None
    
    if limit is None:
        # This is the full trade log page, add filters
        st.subheader("Filter Trades")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Symbol filter
            all_trades = data_service.get_trades()
            all_symbols = sorted(list(set([t['symbol'] for t in all_trades])))
            symbol_filter = st.selectbox(
                "Filter by Symbol",
                ["All Symbols"] + all_symbols
            )
            if symbol_filter == "All Symbols":
                symbol_filter = None
        
        with col2:
            # Strategy filter
            all_strategies = sorted(list(set([t['strategy'] for t in all_trades])))
            strategy_filter = st.selectbox(
                "Filter by Strategy",
                ["All Strategies"] + all_strategies
            )
            if strategy_filter == "All Strategies":
                strategy_filter = None
        
        with col3:
            # Date filter
            date_options = [
                "All Time",
                "Today",
                "Past 3 Days",
                "Past Week",
                "Past Month"
            ]
            date_selection = st.selectbox(
                "Filter by Date",
                date_options
            )
            
            # Convert selection to date filter
            if date_selection != "All Time":
                today = datetime.datetime.now().date()
                if date_selection == "Today":
                    date_filter = today
                elif date_selection == "Past 3 Days":
                    date_filter = today - datetime.timedelta(days=3)
                elif date_selection == "Past Week":
                    date_filter = today - datetime.timedelta(days=7)
                elif date_selection == "Past Month":
                    date_filter = today - datetime.timedelta(days=30)
    
    # Get trades with filters applied
    trades = data_service.get_trades(limit=limit, symbol=symbol_filter, strategy=strategy_filter)
    
    # Apply date filter if needed
    if date_filter:
        trades = [
            t for t in trades 
            if t['timestamp'].date() >= date_filter
        ]
    
    # Display trades
    if trades:
        # Create a DataFrame from trades
        trades_df = pd.DataFrame(trades)
        
        # Format columns for display
        trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'price' in trades_df.columns:
            trades_df['price'] = trades_df['price'].map('${:,.2f}'.format)
        if 'pnl' in trades_df.columns:
            # Format PnL if it's not None
            trades_df['pnl'] = trades_df['pnl'].apply(
                lambda x: '${:,.2f}'.format(x) if x is not None else 'Open'
            )
            
        # Add a formatted column for action
        if 'action' in trades_df.columns:
            trades_df['action_formatted'] = trades_df['action'].apply(
                lambda x: f"🟢 {x}" if x == "BUY" else f"🔴 {x}"
            )
            
            # Ensure we keep the original action column for sorting
            column_order = list(trades_df.columns)
            action_idx = column_order.index('action')
            column_order.insert(action_idx, 'action_formatted')
            column_order.remove('action')
            trades_df = trades_df[column_order]
            trades_df = trades_df.rename(columns={'action_formatted': 'action'})
        
        # Display as interactive dataframe
        st.dataframe(
            trades_df,
            column_config={
                "timestamp": "Time",
                "symbol": "Symbol",
                "action": "Action",
                "quantity": "Quantity",
                "price": "Price",
                "strategy": "Strategy",
                "pnl": "P&L"
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Summary statistics if in full view
        if limit is None:
            # If we have real trades (not just mock data), calculate summary stats
            st.subheader("Trade Summary")
            
            # Calculate statistics
            total_trades = len(trades)
            buy_trades = sum(1 for t in trades if t['action'] == "🟢 BUY" or t['action'] == "BUY")
            sell_trades = sum(1 for t in trades if t['action'] == "🔴 SELL" or t['action'] == "SELL")
            
            # Count profitable trades (if PnL is available)
            profitable_trades = sum(1 for t in trades 
                                  if t.get('pnl') is not None and 
                                  (isinstance(t['pnl'], (int, float)) and t['pnl'] > 0 or
                                   isinstance(t['pnl'], str) and float(t['pnl'].replace('$', '').replace(',', '')) > 0))
            
            # Calculate win rate if we have closed trades with PnL
            closed_trades = sum(1 for t in trades if t.get('pnl') is not None and t['pnl'] != 'Open')
            win_rate = (profitable_trades / closed_trades * 100) if closed_trades > 0 else 0
            
            # Display statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", total_trades)
            
            with col2:
                st.metric("Buy Trades", buy_trades)
            
            with col3:
                st.metric("Sell Trades", sell_trades)
            
            with col4:
                st.metric("Win Rate", f"{win_rate:.2f}%")
    else:
        st.info("No trades match the current filters" if limit is None else "No trades available")
