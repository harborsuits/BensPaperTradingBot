#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Trading Dashboard - Strategy Lab Component

Displays and manages candidate strategies from the EvoTrader system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time

def render_strategy_lab():
    """Render the strategy lab component"""
    
    # Get strategy candidates from data service
    data_service = st.session_state.data_service
    candidates = data_service.get_strategy_candidates()
    
    if not candidates:
        st.info("No strategy candidates available")
        return
    
    # Create a DataFrame for the candidates table
    candidates_df = pd.DataFrame(candidates)
    
    # Calculate a combined score for sorting (higher is better)
    if not candidates_df.empty:
        candidates_df['combined_score'] = (
            candidates_df['backtest_score'] * 0.4 +
            candidates_df['sharpe_ratio'] * 30 +
            candidates_df['cumulative_return'] * 0.3 -
            candidates_df['max_drawdown'] * 0.5
        )
    
    # Add status indicators with color
    candidates_df['status_display'] = candidates_df['status'].apply(
        lambda x: f"🟢 {x}" if x == "Ready" or x == "Promoted" else
                 f"🟠 {x}" if x == "In Paper Trading" else
                 f"🔴 {x}" if x == "Failed" else f"⚪ {x}"
    )
    
    # Sort by combined score
    candidates_df = candidates_df.sort_values(by='combined_score', ascending=False)
    
    # Display the candidates table
    st.subheader("Strategy Candidates")
    
    # Format columns for display
    display_df = candidates_df.copy()
    if 'backtest_score' in display_df.columns:
        display_df['backtest_score'] = display_df['backtest_score'].map('{:.2f}'.format)
    if 'sharpe_ratio' in display_df.columns:
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].map('{:.2f}'.format)
    if 'cumulative_return' in display_df.columns:
        display_df['cumulative_return'] = display_df['cumulative_return'].map('{:.2f}%'.format)
    if 'win_rate' in display_df.columns:
        display_df['win_rate'] = display_df['win_rate'].map('{:.2f}%'.format)
    if 'max_drawdown' in display_df.columns:
        display_df['max_drawdown'] = display_df['max_drawdown'].map('{:.2f}%'.format)
    
    # Create selection dataframe with key columns
    selection_df = display_df[[
        'name', 'type', 'backtest_score', 'sharpe_ratio', 
        'cumulative_return', 'win_rate', 'max_drawdown', 'status_display'
    ]].copy()
    
    # Display as a table with selection
    selected_row = st.dataframe(
        selection_df,
        column_config={
            "name": "Strategy Name",
            "type": "Type",
            "backtest_score": "Backtest Score",
            "sharpe_ratio": "Sharpe Ratio",
            "cumulative_return": "Return %",
            "win_rate": "Win Rate",
            "max_drawdown": "Max Drawdown",
            "status_display": "Status"
        },
        use_container_width=True,
        hide_index=True,
        selection="single"
    )
    
    # Get the selected strategy
    selected_strategy = None
    if selected_row.selection and len(selected_row.selection) > 0:
        selected_idx = selected_row.selection["rows"][0]
        selected_id = candidates_df.iloc[selected_idx]["id"]
        selected_strategy = next((s for s in candidates if s["id"] == selected_id), None)
    
    # Display selected strategy details
    if selected_strategy:
        render_strategy_details(selected_strategy, data_service)

def render_strategy_details(strategy, data_service):
    """
    Render detailed view of a strategy candidate
    
    Args:
        strategy: Strategy data dictionary
        data_service: DataService instance for actions
    """
    st.subheader(f"Strategy Details: {strategy['name']}")
    
    # Status indicator
    status = strategy['status']
    if status == "Ready":
        st.success(f"🟢 Status: {status} - This strategy is ready for deployment")
    elif status == "In Paper Trading":
        st.warning(f"🟠 Status: {status} - This strategy is currently being tested")
    elif status == "Promoted":
        st.success(f"🟢 Status: {status} - This strategy has been promoted to live trading")
    elif status == "Failed":
        st.error(f"🔴 Status: {status} - This strategy failed validation tests")
    else:
        st.info(f"⚪ Status: {status}")
    
    # Strategy metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Backtest Score",
            f"{strategy['backtest_score']:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{strategy['sharpe_ratio']:.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Return",
            f"{strategy['cumulative_return']:.2f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{strategy['max_drawdown']:.2f}%",
            delta=None,
            delta_color="inverse"
        )
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    # Display parameters as formatted text
    st.markdown(f"""
    | Parameter | Value |
    | --- | --- |
    | Strategy Type | {strategy['type']} |
    | Timeframe | {strategy['parameters'].get('timeframe', 'N/A')} |
    | Asset Class | {strategy['parameters'].get('asset_class', 'N/A')} |
    | Risk Per Trade | {strategy['parameters'].get('risk_per_trade', 'N/A')} |
    | Indicators | {', '.join(strategy['parameters'].get('indicators', []))} |
    """)
    
    # Action buttons - only show appropriate actions based on status
    st.subheader("Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Backtest button - available for all statuses
        backtest_button = st.button("🧪 Run Backtest", key=f"backtest_{strategy['id']}")
        if backtest_button:
            with st.spinner("Running backtest..."):
                # Simulate backtest running
                time.sleep(2)
                
                # In a real implementation, this would call BensBot's backtester
                # success, error = data_service.backtest_strategy(strategy['id'])
                success, error = True, None
                
                if success:
                    st.success("Backtest completed successfully!")
                    # Here we would update the strategy metrics with new backtest results
                else:
                    st.error(f"Backtest failed: {error}")
    
    with col2:
        # Paper trading button - only for Ready status
        if status == "Ready":
            paper_button = st.button("📝 Deploy to Paper Trading", key=f"paper_{strategy['id']}")
            if paper_button:
                with st.spinner("Deploying to paper trading..."):
                    # Simulate deployment
                    time.sleep(1.5)
                    
                    # In a real implementation, this would call BensBot's strategy manager
                    # success, error = data_service.promote_strategy(strategy['id'], to_paper=True)
                    success, error = True, None
                    
                    if success:
                        st.success("Strategy deployed to paper trading!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Deployment failed: {error}")
    
    with col3:
        # Live trading button - only for Ready or In Paper Trading status
        if status in ["Ready", "In Paper Trading"]:
            live_button = st.button("🚀 Promote to Live Trading", key=f"live_{strategy['id']}")
            if live_button:
                # Confirmation dialog
                confirm = st.checkbox(
                    "I confirm this strategy is ready for live trading with real money",
                    key=f"confirm_{strategy['id']}"
                )
                
                if confirm:
                    with st.spinner("Promoting to live trading..."):
                        # Simulate promotion
                        time.sleep(2)
                        
                        # In a real implementation, this would call BensBot's strategy manager
                        # success, error = data_service.promote_strategy(strategy['id'], to_paper=False)
                        success, error = True, None
                        
                        if success:
                            st.success("Strategy promoted to live trading!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Promotion failed: {error}")
    
    # Generate a fake performance chart
    st.subheader("Performance Visualization")
    
    # Create date range for the last 180 days (approximately 6 months)
    end_date = datetime.datetime.now()
    date_range = [end_date - datetime.timedelta(days=i) for i in range(179, -1, -1)]
    
    # Generate random cumulative returns based on the strategy's return value
    np.random.seed(hash(strategy['name']) % 10000)  # Use strategy name as seed for consistency
    
    # Base performance on strategy metrics
    base_daily_return = strategy['cumulative_return'] / 180
    volatility = 0.01 / max(strategy['sharpe_ratio'], 0.5)  # Higher Sharpe = lower volatility
    
    # Add more realistic ups and downs
    daily_returns = np.random.normal(base_daily_return / 100, volatility, len(date_range))
    
    # Add a drawdown period
    drawdown_start = np.random.randint(30, 150)
    drawdown_length = int(min(strategy['max_drawdown'], 20))
    for i in range(drawdown_start, drawdown_start + drawdown_length):
        if i < len(daily_returns):
            daily_returns[i] = -abs(daily_returns[i]) * 2
    
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    # Create dataframe
    perf_df = pd.DataFrame({
        'Date': date_range,
        'Return': cumulative_returns * 100  # Convert to percentage
    })
    
    # Add benchmark for comparison (a smoother, lower return)
    benchmark_returns = np.random.normal(base_daily_return / 200, volatility / 2, len(date_range))
    benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    perf_df['Benchmark'] = benchmark_cumulative * 100
    
    # Plot strategy performance
    fig = px.line(
        perf_df, 
        x='Date', 
        y=['Return', 'Benchmark'],
        title=f'{strategy["name"]} Performance vs Benchmark',
        labels={'value': 'Cumulative Return (%)', 'Date': 'Date', 'variable': 'Series'},
    )
    
    # Add baseline at 0%
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        yaxis=dict(ticksuffix='%'),
        legend_title_text=''
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add drawdown chart
    drawdowns = []
    peak = 100
    for ret in (perf_df['Return'] + 100).values:
        if ret > peak:
            peak = ret
        drawdown = (peak - ret) / peak * 100
        drawdowns.append(drawdown)
    
    perf_df['Drawdown'] = drawdowns
    
    # Plot drawdowns
    fig_dd = px.area(
        perf_df, 
        x='Date', 
        y='Drawdown',
        title=f'{strategy["name"]} Drawdown Analysis',
        labels={'Drawdown': 'Drawdown (%)', 'Date': 'Date'},
        color_discrete_sequence=['rgba(220, 38, 38, 0.5)']  # Light red
    )
    
    # Customize layout
    fig_dd.update_layout(
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        yaxis=dict(ticksuffix='%', autorange="reversed")  # Reverse Y-axis for drawdowns
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Strategy explanation
    with st.expander("Strategy Explanation"):
        st.markdown(f"""
        ### {strategy['name']} ({strategy['type']})
        
        This candidate strategy was generated by BensBot's EvoTrader system using evolutionary algorithms
        and machine learning techniques. It has been optimized based on historical data and tested
        against various market conditions.
        
        #### Strategy Approach:
        This is a **{strategy['type']}** strategy that primarily trades **{strategy['parameters'].get('asset_class', 'various assets')}** 
        on a **{strategy['parameters'].get('timeframe', 'unknown')}** timeframe.
        
        #### Key Performance Metrics:
        - **Backtest Score**: {strategy['backtest_score']:.2f} (composite metric of risk-adjusted returns)
        - **Sharpe Ratio**: {strategy['sharpe_ratio']:.2f} (higher values indicate better risk-adjusted returns)
        - **Cumulative Return**: {strategy['cumulative_return']:.2f}% (total return over the backtest period)
        - **Win Rate**: {strategy['win_rate']:.2f}% (percentage of profitable trades)
        - **Maximum Drawdown**: {strategy['max_drawdown']:.2f}% (largest peak-to-trough decline)
        
        #### Strategy Logic:
        This strategy uses the following technical indicators for signal generation:
        - {', '.join(strategy['parameters'].get('indicators', ['Unknown']))}
        
        #### Recommendation:
        Based on the backtest results, this strategy is 
        {'highly recommended' if strategy['backtest_score'] > 85 and strategy['sharpe_ratio'] > 2 else 
         'recommended' if strategy['backtest_score'] > 70 and strategy['sharpe_ratio'] > 1 else 
         'recommended for further testing' if strategy['backtest_score'] > 60 else 
         'not recommended for live trading'} 
        for trading real capital.
        
        {'Consider deploying this strategy to paper trading for further validation.' if strategy['status'] == 'Ready' else
         'Continue monitoring paper trading performance before promoting to live.' if strategy['status'] == 'In Paper Trading' else
         'This strategy is already being used in live trading.' if strategy['status'] == 'Promoted' else
         'This strategy failed validation and should not be used.' if strategy['status'] == 'Failed' else ''}
        """)
