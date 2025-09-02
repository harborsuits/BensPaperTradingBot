"""
Broker Balances Component

This component displays broker account balances and open positions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_broker_balances(data_service, simplified: bool = False):
    """
    Render the broker balances and positions component.
    
    Args:
        data_service: Data service for fetching broker data
        simplified: Whether to show a simplified view
    """
    # Get broker account data
    broker_df = data_service.get_broker_balances()
    
    # Get positions data
    positions_df = data_service.get_positions()
    
    # Check if we have data
    if broker_df.empty:
        st.warning("No broker account data available.")
        return
    
    # For simplified view, just show key metrics
    if simplified:
        # Show broker accounts in a simple table
        display_df = broker_df.copy()
        
        # Format values
        if 'cash' in display_df.columns:
            display_df['cash'] = display_df['cash'].apply(lambda x: f"${x:,.2f}")
        
        if 'equity' in display_df.columns:
            display_df['equity'] = display_df['equity'].apply(lambda x: f"${x:,.2f}")
        
        if 'daily_pnl' in display_df.columns:
            # Color code P&L
            display_df['daily_pnl_display'] = display_df['daily_pnl'].apply(
                lambda x: f"<span style='color:{'#4CAF50' if x > 0 else '#F44336' if x < 0 else '#757575'};'>${x:,.2f}</span>"
            )
            display_df = display_df.drop('daily_pnl', axis=1)
        
        # Rename columns
        column_renames = {
            'name': 'Broker',
            'type': 'Type',
            'status': 'Status',
            'cash': 'Cash',
            'equity': 'Equity',
            'daily_pnl_display': 'Today P&L'
        }
        
        # Only rename columns that exist
        rename_dict = {k: v for k, v in column_renames.items() if k in display_df.columns}
        display_df = display_df.rename(columns=rename_dict)
        
        # Drop unnecessary columns
        drop_cols = ['broker_id', 'buying_power', 'daily_pnl_pct']
        display_df = display_df.drop([col for col in drop_cols if col in display_df.columns], axis=1)
        
        # Display the table with HTML formatting
        st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Show positions count if available
        if not positions_df.empty:
            st.caption(f"Total positions: {len(positions_df)}")
    
    # Detailed view
    else:
        # Create tabs
        tabs = st.tabs(["Broker Accounts", "Open Positions", "Portfolio Analysis"])
        
        with tabs[0]:
            st.subheader("Broker Accounts")
            
            # Full broker accounts table
            display_df = broker_df.copy()
            
            # Format values
            if 'cash' in display_df.columns:
                display_df['cash'] = display_df['cash'].apply(lambda x: f"${x:,.2f}")
            
            if 'equity' in display_df.columns:
                display_df['equity'] = display_df['equity'].apply(lambda x: f"${x:,.2f}")
            
            if 'buying_power' in display_df.columns:
                display_df['buying_power'] = display_df['buying_power'].apply(lambda x: f"${x:,.2f}")
            
            if 'daily_pnl' in display_df.columns:
                # Color code P&L
                display_df['daily_pnl'] = display_df['daily_pnl'].apply(
                    lambda x: f"<span style='color:{'#4CAF50' if x > 0 else '#F44336' if x < 0 else '#757575'};'>${x:,.2f}</span>"
                )
            
            if 'daily_pnl_pct' in display_df.columns:
                display_df['daily_pnl_pct'] = display_df['daily_pnl_pct'].apply(
                    lambda x: f"<span style='color:{'#4CAF50' if x > 0 else '#F44336' if x < 0 else '#757575'};'>{x:.2f}%</span>"
                )
            
            # Add status styling
            if 'status' in display_df.columns:
                display_df['status'] = display_df['status'].apply(
                    lambda x: f"<span style='color:{'#4CAF50' if x == 'Connected' else '#F44336'};'>{x}</span>"
                )
            
            # Rename columns
            column_renames = {
                'name': 'Broker',
                'type': 'Type',
                'status': 'Status',
                'cash': 'Cash',
                'equity': 'Equity',
                'buying_power': 'Buying Power',
                'daily_pnl': 'Today P&L',
                'daily_pnl_pct': 'Today P&L %'
            }
            
            # Only rename columns that exist
            rename_dict = {k: v for k, v in column_renames.items() if k in display_df.columns}
            display_df = display_df.rename(columns=rename_dict)
            
            # Drop broker_id
            if 'broker_id' in display_df.columns:
                display_df = display_df.drop('broker_id', axis=1)
            
            # Display the table with HTML formatting
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Broker allocation pie chart
            if 'equity' in broker_df.columns and 'name' in broker_df.columns:
                st.subheader("Capital Allocation by Broker")
                
                # Create a pie chart of equity allocation
                fig = px.pie(
                    broker_df,
                    values='equity',
                    names='name',
                    title="Equity Distribution",
                    hole=0.4
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.subheader("Open Positions")
            
            if positions_df.empty:
                st.info("No open positions.")
            else:
                # Display open positions
                display_df = positions_df.copy()
                
                # Format values
                if 'entry_price' in display_df.columns:
                    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:,.2f}")
                
                if 'current_price' in display_df.columns:
                    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
                
                if 'market_value' in display_df.columns:
                    display_df['market_value'] = display_df['market_value'].apply(lambda x: f"${x:,.2f}")
                
                if 'unrealized_pnl' in display_df.columns:
                    # Color code P&L
                    display_df['unrealized_pnl'] = display_df['unrealized_pnl'].apply(
                        lambda x: f"<span style='color:{'#4CAF50' if x > 0 else '#F44336' if x < 0 else '#757575'};'>${x:,.2f}</span>"
                    )
                
                if 'unrealized_pnl_pct' in display_df.columns:
                    display_df['unrealized_pnl_pct'] = display_df['unrealized_pnl_pct'].apply(
                        lambda x: f"<span style='color:{'#4CAF50' if x > 0 else '#F44336' if x < 0 else '#757575'};'>{x:.2f}%</span>"
                    )
                
                # Add side styling
                if 'side' in display_df.columns:
                    display_df['side'] = display_df['side'].apply(
                        lambda x: f"<span style='color:{'#4CAF50' if x == 'LONG' else '#F44336'};'>{x}</span>"
                    )
                
                # Rename columns
                column_renames = {
                    'broker': 'Broker',
                    'strategy': 'Strategy',
                    'symbol': 'Symbol',
                    'quantity': 'Quantity',
                    'side': 'Side',
                    'entry_price': 'Entry',
                    'current_price': 'Current',
                    'market_value': 'Value',
                    'unrealized_pnl': 'Unrealized P&L',
                    'unrealized_pnl_pct': 'P&L %'
                }
                
                # Only rename columns that exist
                rename_dict = {k: v for k, v in column_renames.items() if k in display_df.columns}
                display_df = display_df.rename(columns=rename_dict)
                
                # Display the table with HTML formatting
                st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Position analysis
                st.subheader("Position Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Position by symbol (market value)
                    if all(col in positions_df.columns for col in ['symbol', 'market_value']):
                        # Group by symbol and sum market value
                        symbol_value = positions_df.groupby('symbol')['market_value'].sum().reset_index()
                        symbol_value = symbol_value.sort_values('market_value', ascending=False)
                        
                        fig = px.bar(
                            symbol_value,
                            x='symbol',
                            y='market_value',
                            title="Position Size by Symbol",
                            color_discrete_sequence=['#2196F3']
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title=None,
                            yaxis_title="Market Value ($)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # P&L by symbol
                    if all(col in positions_df.columns for col in ['symbol', 'unrealized_pnl']):
                        # Group by symbol and sum P&L
                        symbol_pnl = positions_df.groupby('symbol')['unrealized_pnl'].sum().reset_index()
                        symbol_pnl = symbol_pnl.sort_values('unrealized_pnl', ascending=False)
                        
                        # Create P&L bar chart with green/red coloring
                        fig = go.Figure()
                        
                        for idx, row in symbol_pnl.iterrows():
                            fig.add_trace(go.Bar(
                                x=[row['symbol']],
                                y=[row['unrealized_pnl']],
                                name=row['symbol'],
                                marker_color='#4CAF50' if row['unrealized_pnl'] >= 0 else '#F44336'
                            ))
                        
                        fig.update_layout(
                            title="P&L by Symbol",
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title=None,
                            yaxis_title="P&L ($)",
                            showlegend=False,
                            barmode='relative'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.subheader("Portfolio Analysis")
            
            # Portfolio allocation visualizations
            if not positions_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Allocation by strategy
                    if all(col in positions_df.columns for col in ['strategy', 'market_value']):
                        # Group by strategy and sum market value
                        strategy_value = positions_df.groupby('strategy')['market_value'].sum().reset_index()
                        
                        fig = px.pie(
                            strategy_value,
                            values='market_value',
                            names='strategy',
                            title="Allocation by Strategy",
                            hole=0.4
                        )
                        
                        fig.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Long vs Short allocation
                    if all(col in positions_df.columns for col in ['side', 'market_value']):
                        # Group by side and sum market value
                        side_value = positions_df.groupby('side')['market_value'].sum().reset_index()
                        
                        fig = px.pie(
                            side_value,
                            values='market_value',
                            names='side',
                            title="Long vs Short Exposure",
                            color='side',
                            color_discrete_map={
                                'LONG': '#4CAF50',
                                'SHORT': '#F44336'
                            },
                            hole=0.4
                        )
                        
                        fig.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display portfolio metrics
                st.subheader("Portfolio Metrics")
                
                # Portfolio summary data
                portfolio_summary = data_service.get_portfolio_summary()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Equity",
                        f"${portfolio_summary.get('total_equity', 0):,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Cash",
                        f"${portfolio_summary.get('total_cash', 0):,.2f}",
                        f"{portfolio_summary.get('total_cash', 0) / portfolio_summary.get('total_equity', 1) * 100:.1f}% of equity"
                    )
                
                with col3:
                    st.metric(
                        "Invested",
                        f"${portfolio_summary.get('total_invested', 0):,.2f}",
                        f"{portfolio_summary.get('total_invested', 0) / portfolio_summary.get('total_equity', 1) * 100:.1f}% of equity"
                    )
                
                with col4:
                    st.metric(
                        "Total P&L",
                        f"${portfolio_summary.get('total_pnl', 0):,.2f}",
                        f"{portfolio_summary.get('total_pnl_pct', 0):.2f}%",
                        delta_color="normal"
                    )
            else:
                st.info("No open positions to analyze.")
