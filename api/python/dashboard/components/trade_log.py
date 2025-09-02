"""
Trade Log Component

This component displays a log of all trades, with filtering and sorting options.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

def render_trade_log(data_service, max_trades: int = 100, with_filters: bool = False, account_type: str = None):
    """
    Render the trade log component.
    
    Args:
        data_service: Data service for fetching trade data
        max_trades: Maximum number of trades to display
        with_filters: Whether to show filtering options
    """
    # Get strategy list for filter
    strategies_df = data_service.get_active_strategies()
    strategy_names = []
    if not strategies_df.empty and 'name' in strategies_df.columns:
        strategy_names = list(strategies_df['name'].unique())
    
    # Apply filters if requested
    strategy_filter = None
    
    if with_filters:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if strategy_names:
                strategy_filter = st.selectbox(
                    "Filter by Strategy",
                    options=["All Strategies"] + strategy_names,
                    index=0
                )
                if strategy_filter == "All Strategies":
                    strategy_filter = None
        
        with col2:
            # Date range filter would go here
            # For now, we'll skip implementing this as the mock data doesn't have real dates
            st.selectbox(
                "Date Range",
                options=["Today", "Yesterday", "Last 7 Days", "This Month", "All Time"],
                index=4
            )
            
        with col3:
            # Refresh button
            if st.button("Refresh Data", key="refresh_trades"):
                st.rerun()
    
    # Get trade log data with filter
    trades_df = data_service.get_trade_log(
        max_trades=max_trades,
        strategy_filter=strategy_filter,
        account_type=account_type
    )
    
    if trades_df.empty:
        st.warning("No trades found in the log.")
        return
    
    # Format the DataFrame for display
    display_df = trades_df.copy()
    
    # Format timestamp
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df['timestamp'] = display_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Format P&L with colors
    if 'pnl' in display_df.columns:
        # Convert to string with styling
        display_df['pnl_display'] = display_df['pnl'].apply(
            lambda x: f"${x:.2f}" if x is not None else "—"
        )
        display_df['pnl_style'] = display_df['pnl'].apply(
            lambda x: "positive" if x is not None and x > 0 else
                      "negative" if x is not None and x < 0 else ""
        )
    
    # Format prices
    if 'price' in display_df.columns:
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    
    # Format quantity based on symbol
    if 'quantity' in display_df.columns and 'symbol' in display_df.columns:
        display_df['quantity'] = display_df.apply(
            lambda row: f"{row['quantity']:.4f}" if 'USD' in str(row['symbol']) else f"{row['quantity']:,}",
            axis=1
        )
    
    # Add status styling
    if 'status' in display_df.columns:
        display_df['status_style'] = display_df['status'].apply(
            lambda x: "success" if x == "Filled" else
                      "warning" if x == "Partially Filled" else
                      "danger" if x == "Cancelled" else ""
        )
    
    # Display the trade log with styling
    if 'pnl' in display_df.columns:
        # Add P&L styling
        format_dict = {}
        if 'pnl_style' in display_df.columns:
            def _format_pnl(row):
                if row['pnl_style'] == 'positive':
                    return f"<span style='color: #4CAF50; font-weight: bold;'>{row['pnl_display']}</span>"
                elif row['pnl_style'] == 'negative':
                    return f"<span style='color: #F44336; font-weight: bold;'>{row['pnl_display']}</span>"
                else:
                    return row['pnl_display']
            
            display_df['P&L'] = display_df.apply(_format_pnl, axis=1)
            display_df = display_df.drop(['pnl', 'pnl_display', 'pnl_style'], axis=1)
    
    # Add status styling
    if 'status_style' in display_df.columns:
        def _format_status(row):
            if row['status_style'] == 'success':
                return f"<span style='color: #4CAF50;'>{row['status']}</span>"
            elif row['status_style'] == 'warning':
                return f"<span style='color: #FF9800;'>{row['status']}</span>"
            elif row['status_style'] == 'danger':
                return f"<span style='color: #F44336;'>{row['status']}</span>"
            else:
                return row['status']
        
        display_df['Status'] = display_df.apply(_format_status, axis=1)
        display_df = display_df.drop(['status', 'status_style'], axis=1)
    
    # Rename columns
    column_renames = {
        'timestamp': 'Time',
        'strategy': 'Strategy',
        'symbol': 'Symbol',
        'action': 'Action',
        'quantity': 'Size',
        'price': 'Price'
    }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in column_renames.items() if k in display_df.columns}
    display_df = display_df.rename(columns=rename_dict)
    
    # Reorder columns
    desired_order = ['Time', 'Strategy', 'Symbol', 'Action', 'Size', 'Price', 'Status', 'P&L']
    existing_cols = [col for col in desired_order if col in display_df.columns]
    other_cols = [col for col in display_df.columns if col not in desired_order]
    display_df = display_df[existing_cols + other_cols]
    
    # Display the table with HTML formatting enabled
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Summary statistics if this is the detailed view
    if with_filters and not trades_df.empty:
        st.subheader("Trade Summary")
        
        # Create some summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Trade count by action
            if 'action' in trades_df.columns:
                action_counts = trades_df['action'].value_counts().reset_index()
                action_counts.columns = ['Action', 'Count']
                
                fig = px.pie(
                    action_counts, 
                    values='Count', 
                    names='Action',
                    color='Action',
                    color_discrete_map={'BUY': '#4CAF50', 'SELL': '#F44336'},
                    hole=0.4
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trade count by status
            if 'status' in trades_df.columns:
                status_counts = trades_df['status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                
                fig = px.bar(
                    status_counts,
                    x='Status',
                    y='Count',
                    color='Status',
                    color_discrete_map={
                        'Filled': '#4CAF50',
                        'Partially Filled': '#FF9800',
                        'Cancelled': '#F44336'
                    }
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300,
                    xaxis_title=None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Trade count by symbol
            if 'symbol' in trades_df.columns:
                symbol_counts = trades_df['symbol'].value_counts().reset_index()
                symbol_counts.columns = ['Symbol', 'Count']
                symbol_counts = symbol_counts.head(5)  # Top 5 symbols
                
                fig = px.bar(
                    symbol_counts,
                    x='Symbol',
                    y='Count',
                    title="Top 5 Symbols"
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300,
                    xaxis_title=None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # P&L summary if available
        if 'pnl' in trades_df.columns:
            # Filter out None values
            pnl_df = trades_df[trades_df['pnl'].notna()]
            
            if not pnl_df.empty:
                st.subheader("P&L Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary metrics
                    total_pnl = pnl_df['pnl'].sum()
                    profitable_trades = (pnl_df['pnl'] > 0).sum()
                    losing_trades = (pnl_df['pnl'] < 0).sum()
                    
                    metrics_df = pd.DataFrame({
                        'Metric': [
                            'Total P&L',
                            'Profitable Trades',
                            'Losing Trades',
                            'Win Rate',
                            'Average Win',
                            'Average Loss',
                            'Profit Factor'
                        ],
                        'Value': [
                            f"${total_pnl:.2f}",
                            f"{profitable_trades}",
                            f"{losing_trades}",
                            f"{profitable_trades / len(pnl_df):.1%}" if len(pnl_df) > 0 else "N/A",
                            f"${pnl_df[pnl_df['pnl'] > 0]['pnl'].mean():.2f}" if len(pnl_df[pnl_df['pnl'] > 0]) > 0 else "N/A",
                            f"${pnl_df[pnl_df['pnl'] < 0]['pnl'].mean():.2f}" if len(pnl_df[pnl_df['pnl'] < 0]) > 0 else "N/A",
                            f"{pnl_df[pnl_df['pnl'] > 0]['pnl'].sum() / abs(pnl_df[pnl_df['pnl'] < 0]['pnl'].sum()):.2f}" if abs(pnl_df[pnl_df['pnl'] < 0]['pnl'].sum()) > 0 else "∞"
                        ]
                    })
                    
                    st.table(metrics_df)
                
                with col2:
                    # P&L distribution
                    fig = px.histogram(
                        pnl_df,
                        x='pnl',
                        nbins=20,
                        title="P&L Distribution",
                        color_discrete_sequence=['#2196F3']
                    )
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=300,
                        xaxis_title="P&L ($)",
                        yaxis_title="Count"
                    )
                    # Add a vertical line at zero
                    fig.add_shape(
                        type="line",
                        x0=0, y0=0, x1=0, y1=1,
                        yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    st.plotly_chart(fig, use_container_width=True)
