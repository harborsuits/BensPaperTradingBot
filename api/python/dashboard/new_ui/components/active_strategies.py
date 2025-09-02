#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Active Strategies Component

Displays the currently active trading strategies and their performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

def render_active_strategies():
    """Render the active strategies component"""
    
    # Get active strategies from data service
    data_service = st.session_state.data_service
    strategies = data_service.get_strategies()
    
    if not strategies:
        st.info("No active strategies available")
        return
    
    # Create tabs for each strategy
    tabs = st.tabs([s['name'] for s in strategies])
    
    # Render each strategy in its own tab
    for i, (tab, strategy) in enumerate(zip(tabs, strategies)):
        with tab:
            render_strategy_details(strategy)

def render_strategy_details(strategy):
    """
    Render detailed view of a strategy
    
    Args:
        strategy: Strategy data dictionary
    """
    # Strategy status indicator
    if strategy['status'] == "Running":
        st.success(f"⚡ Status: {strategy['status']}")
    else:
        st.warning(f"⏸️ Status: {strategy['status']}")
    
    # Strategy controls
    col1, col2 = st.columns([1, 5])
    with col1:
        if strategy['status'] == "Running":
            if st.button(f"⏸️ Pause", key=f"pause_{strategy['id']}"):
                # In a real implementation, this would call BensBot's API
                # to pause the strategy
                st.success(f"Strategy {strategy['name']} paused")
                st.rerun()
        else:
            if st.button(f"▶️ Resume", key=f"resume_{strategy['id']}"):
                # In a real implementation, this would call BensBot's API
                # to resume the strategy
                st.success(f"Strategy {strategy['name']} resumed")
                st.rerun()
    
    # Strategy KPIs in columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            "Returns",
            f"{strategy['returns']:.2f}%",
            delta=None
        )
    
    with kpi2:
        st.metric(
            "Sharpe Ratio",
            f"{strategy['sharpe']:.2f}",
            delta=None
        )
    
    with kpi3:
        st.metric(
            "Win Rate",
            f"{strategy['win_rate']:.2f}%",
            delta=None
        )
    
    with kpi4:
        st.metric(
            "# of Trades",
            f"{strategy['trades_count']}",
            delta=None
        )
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    # Display parameters as a table
    params_df = pd.DataFrame({
        'Parameter': list(strategy['parameters'].keys()),
        'Value': list(strategy['parameters'].values())
    })
    
    st.dataframe(
        params_df,
        column_config={
            "Parameter": "Parameter",
            "Value": "Value"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Strategy trades
    st.subheader("Recent Trades")
    
    # Get trades for this strategy
    data_service = st.session_state.data_service
    strategy_trades = data_service.get_trades(strategy=strategy['name'], limit=10)
    
    if strategy_trades:
        # Create a DataFrame from trades
        trades_df = pd.DataFrame(strategy_trades)
        
        # Format columns for display
        trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'price' in trades_df.columns:
            trades_df['price'] = trades_df['price'].map('${:,.2f}'.format)
        if 'pnl' in trades_df.columns:
            # Format PnL if it's not None
            trades_df['pnl'] = trades_df['pnl'].apply(
                lambda x: '${:,.2f}'.format(x) if x is not None else 'Open'
            )
        
        # Display trades
        st.dataframe(
            trades_df,
            column_config={
                "timestamp": "Time",
                "symbol": "Symbol",
                "action": "Action",
                "quantity": "Quantity",
                "price": "Price",
                "pnl": "P&L"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No trades for this strategy yet")
    
    # Performance visualization
    st.subheader("Performance Analysis")
    
    # In a real implementation, we would get actual performance data
    # For now, we'll simulate a performance chart
    
    # Create date range for the last 30 days
    end_date = datetime.datetime.now()
    date_range = [end_date - datetime.timedelta(days=i) for i in range(29, -1, -1)]
    
    # Generate random cumulative returns based on the strategy's return value
    np.random.seed(hash(strategy['name']) % 10000)  # Use strategy name as seed for consistency
    daily_returns = np.random.normal(strategy['returns']/100, 0.03, len(date_range))
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    # Create dataframe
    perf_df = pd.DataFrame({
        'Date': date_range,
        'Return': cumulative_returns * 100  # Convert to percentage
    })
    
    # Plot strategy performance
    fig = px.line(
        perf_df, 
        x='Date', 
        y='Return',
        title=f'{strategy["name"]} Cumulative Returns (%)',
        labels={'Return': 'Cumulative Return (%)', 'Date': 'Date'},
    )
    
    # Add baseline at 0%
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        yaxis=dict(ticksuffix='%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show explanation and logic
    with st.expander("Strategy Logic & Explanation"):
        st.markdown(f"""
        ### {strategy['name']} ({strategy['type']})
        
        This strategy is based on the **{strategy['type']}** approach to trading. It analyzes market conditions 
        and executes trades when specific conditions are met.
        
        #### Key Features:
        - **Timeframe**: {strategy['parameters'].get('timeframe', 'N/A')}
        - **Risk Per Trade**: {strategy['parameters'].get('risk_per_trade', 'N/A')}
        - **Win Rate**: {strategy['win_rate']:.2f}%
        - **Average Trade Duration**: {strategy.get('avg_trade_duration', 'N/A')} minutes
        
        #### Strategy Logic:
        The strategy identifies opportunities based on {strategy['type'].lower()} patterns in the market. 
        When a pattern is detected, it generates buy or sell signals and executes trades accordingly.
        
        For {strategy['type']} strategies, the key indicators typically include:
        - Price momentum
        - Volume analysis
        - Technical indicators
        - Market trends
        
        #### Performance Factors:
        The current Sharpe ratio of {strategy['sharpe']:.2f} indicates 
        {'good' if strategy['sharpe'] > 1.5 else 'moderate' if strategy['sharpe'] > 0.8 else 'poor'} 
        risk-adjusted returns. The strategy has executed {strategy['trades_count']} trades with a 
        win rate of {strategy['win_rate']:.2f}%.
        """)
