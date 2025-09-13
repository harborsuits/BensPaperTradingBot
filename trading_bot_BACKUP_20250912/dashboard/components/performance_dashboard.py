"""
Strategy Performance Dashboard Component

This component displays performance metrics for all strategies, including:
- Performance metrics table
- Equity curves
- Key statistics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def render_performance_dashboard(data_service, time_period: str = "All Time", simplified: bool = False, account_type: str = None):
    """
    Render the strategy performance dashboard.
    
    Args:
        data_service: Data service for fetching performance data
        time_period: Time period for data (Today, This Week, etc.)
        simplified: Whether to show a simplified version
        account_type: Type of account to filter performance data by
    """
    # Get performance data filtered by account type with error handling
    try:
        performance_df = data_service.get_strategy_performance(time_period=time_period, account_type=account_type)
        
        if performance_df.empty:
            st.warning("No strategy performance data available.")
            return
    except AttributeError:
        # Handle missing method by creating mock data
        import pandas as pd
        performance_df = pd.DataFrame([
            {
                'strategy_id': 'default',
                'strategy_name': 'Default Strategy',
                'cumulative_return': 0.05,  # 5%
                'win_rate': 55.0,  # 55%
                'profit_factor': 1.2,
                'sharpe_ratio': 0.8,
                'max_drawdown': -0.03,  # 3%
                'trade_count': 25,
                'avg_trade_duration': '2.5 days'
            }
        ])
        st.info("Using demo strategy performance data")
    
    # Add color coding for strategy phase with dark text for contrast
    if 'phase' in performance_df.columns:
        performance_df['phase_color'] = performance_df['phase'].apply(
            lambda x: "🟢 LIVE" if x == "LIVE" else "🔵 PAPER" if x == "PAPER_TRADE" else "⚪ OTHER"
        )
        
    # Add custom styling for tables to ensure good contrast
    st.markdown("""
    <style>
    /* Ensure proper contrast in tables */
    table {
        color: black !important;
        background-color: white !important;
    }
    th {
        background-color: #f0f0f0 !important;
        color: black !important;
        font-weight: bold !important;
    }
    td {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get equity curves
    equity_curves = data_service.get_strategy_equity_curves()
    
    # For simplified view, just show key metrics and small chart
    if simplified:
        # Show a simplified table with only key metrics
        simple_cols = ['name', 'phase_color', 'pnl', 'pnl_pct', 'sharpe_ratio', 'win_rate']
        if all(col in performance_df.columns for col in simple_cols):
            display_df = performance_df[simple_cols].copy()
            
            # Format values with color coding for proper contrast
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
            display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.0%}")
            
            # Add Streamlit styling to ensure good contrast
            st.markdown("""
            <style>
            /* Ensure proper contrast in dataframes */
            [data-testid="stDataFrame"] table {
                color: black !important;
                background-color: white !important;
            }
            [data-testid="stDataFrame"] th {
                background-color: #f0f0f0 !important;
                color: black !important;
                font-weight: bold !important;
            }
            [data-testid="stDataFrame"] td {
                color: black !important;
                background-color: white !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Rename columns
            display_df.columns = ['Strategy', 'Mode', 'P&L', 'Return', 'Sharpe', 'Win Rate']
            
            # Show the simplified table
            st.dataframe(display_df, use_container_width=True)
            
            # Show a small combined equity curve
            if equity_curves:
                fig = go.Figure()
                
                for strategy_id, curve_df in equity_curves.items():
                    # Find the strategy name
                    strategy_name = performance_df[performance_df['id'] == strategy_id]['name'].values[0] \
                        if 'id' in performance_df.columns and strategy_id in performance_df['id'].values else strategy_id
                    
                    # Add line for each strategy
                    fig.add_trace(go.Scatter(
                        x=curve_df['date'] if 'date' in curve_df.columns else range(len(curve_df)),
                        y=curve_df['equity'] if 'equity' in curve_df.columns else curve_df.iloc[:, 0],
                        mode='lines',
                        name=strategy_name
                    ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title=None,
                    yaxis_title="Equity ($)",
                    template="plotly_white",
                    paper_bgcolor='rgba(255,255,255,1)',  # White background
                    plot_bgcolor='rgba(255,255,255,1)',   # White plot area
                    font=dict(color='black')             # Black text for maximum contrast
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Strategy performance data does not contain required columns.")
    
    # Full dashboard view
    else:
        # Create tabs for different views
        tabs = st.tabs(["Performance Metrics", "Equity Curves", "Return Distribution"])
        
        with tabs[0]:
            # Full metrics table
            st.subheader(f"Strategy Performance - {time_period}")
            
            # Format and display the data
            display_df = performance_df.copy()
            
            # Handle formatting based on available columns
            if 'pnl' in display_df.columns:
                display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            
            if 'pnl_pct' in display_df.columns:
                display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            
            if 'sharpe_ratio' in display_df.columns:
                display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
            
            if 'win_rate' in display_df.columns:
                display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.0%}")
            
            if 'max_drawdown_pct' in display_df.columns:
                display_df['max_drawdown_pct'] = display_df['max_drawdown_pct'].apply(lambda x: f"{x:.2f}%")
            
            if 'profit_factor' in display_df.columns:
                display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")
            
            # Rename columns for display
            column_renames = {
                'name': 'Strategy',
                'phase_color': 'Mode',
                'status': 'Status',
                'pnl': 'P&L',
                'pnl_pct': 'Return',
                'sharpe_ratio': 'Sharpe',
                'win_rate': 'Win Rate',
                'max_drawdown_pct': 'Max DD',
                'trade_count': 'Trades',
                'profit_factor': 'Profit Factor',
                'avg_win': 'Avg Win',
                'avg_loss': 'Avg Loss',
                'expectancy': 'Expectancy'
            }
            
            # Only rename columns that exist
            rename_dict = {k: v for k, v in column_renames.items() if k in display_df.columns}
            display_df = display_df.rename(columns=rename_dict)
            
            # Drop ID column if it exists
            if 'id' in display_df.columns:
                display_df = display_df.drop('id', axis=1)
            
            # Show the full table
            st.dataframe(display_df, use_container_width=True)
            
            # Show key performance metrics by strategy
            if len(performance_df) > 0:
                st.subheader("Key Metrics Comparison")
                
                # Prepare metrics for bar chart
                metrics_to_plot = [
                    ('pnl_pct', 'Return (%)'),
                    ('sharpe_ratio', 'Sharpe Ratio'),
                    ('win_rate', 'Win Rate')
                ]
                
                # Create 3 columns
                cols = st.columns(len(metrics_to_plot))
                
                for i, (metric, title) in enumerate(metrics_to_plot):
                    if metric in performance_df.columns:
                        with cols[i]:
                            fig = px.bar(
                                performance_df,
                                x='name' if 'name' in performance_df.columns else 'id',
                                y=metric,
                                title=title,
                                color='phase' if 'phase' in performance_df.columns else None,
                                color_discrete_map={
                                    'LIVE': '#4CAF50',
                                    'PAPER_TRADE': '#2196F3'
                                }
                            )
                            fig.update_layout(
                                xaxis_title=None,
                                margin=dict(l=0, r=0, t=30, b=0),
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.subheader("Equity Curves")
            
            # Strategy selector
            strategies = list(performance_df['name'].values) if 'name' in performance_df.columns else []
            if strategies:
                selected_strategies = st.multiselect(
                    "Select Strategies to Display",
                    options=strategies,
                    default=strategies[:min(3, len(strategies))]
                )
                
                # Get strategy IDs for selected strategies
                selected_ids = performance_df[performance_df['name'].isin(selected_strategies)]['id'].values \
                    if 'name' in performance_df.columns and 'id' in performance_df.columns else []
                
                # Check if we have equity curves and selected strategies
                # Use len() to safely check if selected_ids has elements
                if equity_curves and len(selected_ids) > 0:
                    # Create equity curve chart
                    fig = go.Figure()
                    
                    for strategy_id in selected_ids:
                        if strategy_id in equity_curves:
                            curve_df = equity_curves[strategy_id]
                            # Safely extract strategy name with proper error handling
                            strategy_names = performance_df[performance_df['id'] == strategy_id]['name'].values
                            strategy_name = strategy_names[0] if len(strategy_names) > 0 else f"Strategy {strategy_id}" \
                                if 'id' in performance_df.columns else strategy_id
                            
                            fig.add_trace(go.Scatter(
                                x=curve_df['date'] if 'date' in curve_df.columns else range(len(curve_df)),
                                y=curve_df['equity'] if 'equity' in curve_df.columns else curve_df.iloc[:, 0],
                                mode='lines',
                                name=strategy_name
                            ))
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Equity ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        template="plotly_white",
                        paper_bgcolor='rgba(255,255,255,1)',  # White background
                        plot_bgcolor='rgba(255,255,255,1)',   # White plot area
                        font=dict(color='black')              # Black text for maximum contrast
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select strategies to view their equity curves.")
            else:
                st.warning("No strategy data available for equity curves.")
        
        with tabs[2]:
            st.subheader("Return Distribution")
            st.info("Return distribution analysis is under development.")
            
            # Placeholder for future return distribution analysis
            # This would typically show histograms of daily/weekly returns
            # and other statistical distributions
