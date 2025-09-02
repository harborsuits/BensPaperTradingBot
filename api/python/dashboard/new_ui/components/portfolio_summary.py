#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Portfolio Summary Component

Displays the current portfolio state, positions, and performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

def render_portfolio_summary():
    """Render the portfolio summary component"""
    
    # Get the latest portfolio data
    data_service = st.session_state.data_service
    portfolio_data = data_service.get_portfolio_data()
    positions = data_service.get_positions()
    status, error = data_service.get_connection_status()
    
    # Connection status and last update time
    st.subheader("Portfolio Status")
    
    # Display connection status
    if status == "Connected":
        st.success(f"✅ Connected to broker")
    else:
        st.error(f"⚠️ {status}: {error if error else 'Unknown error'}")
    
    # Last update time
    if st.session_state.last_update_time:
        last_update = datetime.datetime.fromtimestamp(st.session_state.last_update_time)
        st.info(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display positions table
    st.subheader("Current Positions")
    if positions:
        # Create a DataFrame from positions
        positions_df = pd.DataFrame(positions)
        
        # Format numbers for display
        if not positions_df.empty:
            positions_df['avg_price'] = positions_df['avg_price'].map('${:,.2f}'.format)
            positions_df['current_price'] = positions_df['current_price'].map('${:,.2f}'.format)
            positions_df['unrealized_pnl'] = positions_df['unrealized_pnl'].map('${:,.2f}'.format)
            positions_df['unrealized_pnl_pct'] = positions_df['unrealized_pnl_pct'].map('{:,.2f}%'.format)
        
        # Display as a table
        st.dataframe(
            positions_df,
            column_config={
                "symbol": "Symbol",
                "quantity": "Quantity",
                "avg_price": "Avg. Cost",
                "current_price": "Current Price",
                "unrealized_pnl": "Unrealized P&L",
                "unrealized_pnl_pct": "P&L %",
                "strategy": "Strategy"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No positions currently open")
    
    # Visualize asset allocation if we have positions
    if positions:
        st.subheader("Asset Allocation")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create pie chart for asset allocation
            positions_value = [
                p['quantity'] * float(p['current_price'].replace('$', '').replace(',', '')) 
                if isinstance(p['current_price'], str) 
                else p['quantity'] * p['current_price'] 
                for p in positions
            ]
            symbols = [p['symbol'] for p in positions]
            
            # Create pie chart
            fig = px.pie(
                values=positions_value,
                names=symbols,
                title="Current Allocation by Asset",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show allocation by strategy
            strategy_allocation = {}
            for p in positions:
                strategy = p['strategy']
                position_value = (
                    p['quantity'] * float(p['current_price'].replace('$', '').replace(',', '')) 
                    if isinstance(p['current_price'], str) 
                    else p['quantity'] * p['current_price']
                )
                
                if strategy in strategy_allocation:
                    strategy_allocation[strategy] += position_value
                else:
                    strategy_allocation[strategy] = position_value
            
            # Create DataFrame for display
            strategy_df = pd.DataFrame({
                'Strategy': list(strategy_allocation.keys()),
                'Value': list(strategy_allocation.values())
            })
            
            # Add percentage column
            total_value = strategy_df['Value'].sum()
            strategy_df['Percentage'] = (strategy_df['Value'] / total_value * 100).round(2)
            
            # Format columns
            strategy_df['Value'] = strategy_df['Value'].map('${:,.2f}'.format)
            strategy_df['Percentage'] = strategy_df['Percentage'].map('{:.2f}%'.format)
            
            # Display as a table
            st.dataframe(
                strategy_df,
                column_config={
                    "Strategy": "Strategy",
                    "Value": "Value",
                    "Percentage": "Allocation %"
                },
                hide_index=True,
                use_container_width=True
            )
    
    # Historical portfolio performance
    st.subheader("Portfolio Performance")
    
    # In a real implementation, we would get actual historical data
    # For now, we'll use mock data from the portfolio data
    if 'historical_values' in portfolio_data:
        # Create date range for the last 30 days
        end_date = datetime.datetime.now()
        date_range = [end_date - datetime.timedelta(days=i) for i in range(29, -1, -1)]
        
        # Create dataframe with dates and portfolio values
        hist_df = pd.DataFrame({
            'Date': date_range,
            'Value': portfolio_data['historical_values']
        })
        
        # Plot portfolio value over time
        fig = px.line(
            hist_df, 
            x='Date', 
            y='Value',
            title='Portfolio Value (Last 30 Days)',
            labels={'Value': 'Portfolio Value ($)', 'Date': 'Date'},
        )
        
        # Add today's marker in a different color
        fig.add_trace(
            go.Scatter(
                x=[hist_df['Date'].iloc[-1]],
                y=[hist_df['Value'].iloc[-1]],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Latest'
            )
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            yaxis=dict(tickprefix='$')
        )
        
        st.plotly_chart(fig, use_container_width=True)
