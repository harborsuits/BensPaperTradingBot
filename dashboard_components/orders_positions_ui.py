"""
Orders & Positions UI Rendering Functions for BensBot Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

#############################
# UI Rendering Functions
#############################

def render_positions_table(positions_df, account_type):
    """
    Render interactive positions table with management controls
    """
    st.subheader(f"Current Positions - {account_type}")
    
    if positions_df.empty:
        st.info(f"No positions found for {account_type}")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Total position value
    total_value = positions_df["position_value"].sum()
    with col1:
        st.metric("Total Position Value", f"${total_value:,.2f}")
    
    # Total unrealized P&L
    total_pnl = positions_df["unrealized_pnl"].sum()
    pnl_delta = f"{positions_df['unrealized_pnl_pct'].mean():.2f}%"
    with col2:
        st.metric("Unrealized P&L", f"${total_pnl:,.2f}", delta=pnl_delta)
    
    # Largest position
    largest_pos = positions_df.loc[positions_df["position_value"].idxmax()]
    largest_pct = largest_pos["exposure_pct"]
    with col3:
        st.metric("Largest Position", f"{largest_pos['symbol']}", f"{largest_pct:.2f}% of equity")
    
    # Risk exposure
    potential_loss = positions_df["potential_loss"].sum()
    with col4:
        st.metric("Total Risk Exposure", f"${abs(potential_loss):,.2f}")
    
    # Position filtering options
    col_a, col_b = st.columns([1, 3])
    
    with col_a:
        # Filter by side
        side_filter = st.radio(
            "Position Side",
            ["All", "Long Only", "Short Only"],
            horizontal=True
        )
    
    with col_b:
        # Filter by asset class
        asset_classes = ["All"] + sorted(positions_df["asset_class"].unique().tolist())
        asset_class_filter = st.multiselect(
            "Asset Class",
            options=asset_classes,
            default=["All"]
        )
    
    # Apply filters
    filtered_df = positions_df.copy()
    
    if side_filter == "Long Only":
        filtered_df = filtered_df[filtered_df["side"] == "long"]
    elif side_filter == "Short Only":
        filtered_df = filtered_df[filtered_df["side"] == "short"]
    
    if asset_class_filter and "All" not in asset_class_filter:
        filtered_df = filtered_df[filtered_df["asset_class"].isin(asset_class_filter)]
    
    # Prepare display DataFrame
    display_cols = [
        "symbol", "side", "quantity", "entry_price", "current_price",
        "unrealized_pnl", "unrealized_pnl_pct", "position_value", "exposure_pct"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = "N/A"
    
    display_df = filtered_df[display_cols].copy()
    
    # Format columns
    display_df["side"] = display_df["side"].str.upper()
    display_df["unrealized_pnl"] = display_df["unrealized_pnl"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    display_df["unrealized_pnl_pct"] = display_df["unrealized_pnl_pct"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    display_df["position_value"] = display_df["position_value"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    display_df["exposure_pct"] = display_df["exposure_pct"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    
    # Rename columns for display
    display_df.columns = [
        "Symbol", "Side", "Quantity", "Entry Price", "Current Price",
        "Unrealized P&L", "% Change", "Position Value", "% of Portfolio"
    ]
    
    # Add color coding for P&L
    def color_pnl(val):
        try:
            # Extract numeric value from string (e.g. "$123.45" -> 123.45)
            if isinstance(val, str) and val.startswith("$"):
                num_val = float(val.replace("$", "").replace(",", ""))
            elif isinstance(val, str) and val.endswith("%"):
                num_val = float(val.replace("%", ""))
            else:
                num_val = float(val)
            
            if num_val > 0:
                return 'color: green'
            elif num_val < 0:
                return 'color: red'
            return ''
        except:
            return ''
    
    # Style the dataframe
    styled_df = display_df.style.applymap(
        color_pnl, 
        subset=["Unrealized P&L", "% Change"]
    )
    
    # Display the table with positions
    st.dataframe(styled_df, use_container_width=True)
    
    # Position management section
    st.subheader("Position Management")
    
    # Position selection for detailed view
    selected_symbol = st.selectbox(
        "Select a position to manage",
        [""] + filtered_df["symbol"].tolist()
    )
    
    # Show position details and controls if a position is selected
    if selected_symbol:
        selected_position = filtered_df[filtered_df["symbol"] == selected_symbol].iloc[0]
        
        # Get position history data (either real or mock)
        position_history = get_position_history(None, selected_symbol, account_type)
        
        # Render detailed view
        render_position_details(selected_position, position_history)
        
        # Position management controls
        st.subheader("Position Controls")
        
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            if st.button(f"Close {selected_symbol} Position", key=f"close_{selected_position['symbol']}"):
                st.warning(f"This would close the {selected_symbol} position in a real implementation")
        
        with col_y:
            if st.button(f"Modify Stop Loss", key=f"modify_sl_{selected_position['symbol']}"):
                st.info(f"This would modify the stop loss for {selected_symbol} in a real implementation")
        
        with col_z:
            if st.button(f"Modify Take Profit", key=f"modify_tp_{selected_position['symbol']}"):
                st.info(f"This would modify the take profit for {selected_symbol} in a real implementation")

def render_open_orders_table(orders_df, account_type):
    """
    Render open orders table with management controls
    """
    st.subheader(f"Open Orders - {account_type}")
    
    if orders_df.empty:
        st.info(f"No open orders found for {account_type}")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    # Total orders
    with col1:
        st.metric("Total Open Orders", f"{len(orders_df)}")
    
    # Total order value
    total_value = orders_df["order_value"].sum()
    with col2:
        st.metric("Total Order Value", f"${total_value:,.2f}")
    
    # Orders by status
    status_counts = orders_df["status"].value_counts()
    primary_status = status_counts.index[0] if not status_counts.empty else "None"
    primary_count = status_counts.iloc[0] if not status_counts.empty else 0
    with col3:
        st.metric(f"Orders by Status", f"{primary_status.title()}: {primary_count}")
    
    # Order filtering options
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Filter by side
        side_filter = st.radio(
            "Order Side",
            ["All", "Buy Only", "Sell Only"],
            horizontal=True
        )
    
    with col_b:
        # Filter by order type
        order_types = ["All"] + sorted(orders_df["order_type"].unique().tolist())
        order_type_filter = st.multiselect(
            "Order Type",
            options=order_types,
            default=["All"]
        )
    
    # Apply filters
    filtered_df = orders_df.copy()
    
    if side_filter == "Buy Only":
        filtered_df = filtered_df[filtered_df["side"] == "buy"]
    elif side_filter == "Sell Only":
        filtered_df = filtered_df[filtered_df["side"] == "sell"]
    
    if order_type_filter and "All" not in order_type_filter:
        filtered_df = filtered_df[filtered_df["order_type"].isin(order_type_filter)]
    
    # Prepare display DataFrame
    display_cols = [
        "symbol", "side", "quantity", "order_type", "limit_price", "stop_price",
        "current_price", "status", "time_in_force", "minutes_active"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = "N/A"
    
    display_df = filtered_df[display_cols].copy()
    
    # Format columns
    display_df["side"] = display_df["side"].str.upper()
    display_df["order_type"] = display_df["order_type"].str.replace("_", " ").str.title()
    display_df["status"] = display_df["status"].str.replace("_", " ").str.title()
    
    # Format time active
    def format_time_active(minutes):
        if not isinstance(minutes, (int, float)):
            return minutes
        
        hours = minutes // 60
        mins = minutes % 60
        
        if hours > 0:
            return f"{hours}h {mins}m"
        else:
            return f"{mins}m"
    
    display_df["minutes_active"] = display_df["minutes_active"].apply(format_time_active)
    
    # Rename columns for display
    display_df.columns = [
        "Symbol", "Side", "Quantity", "Order Type", "Limit Price", "Stop Price",
        "Current Price", "Status", "Time in Force", "Time Active"
    ]
    
    # Add color coding for order status
    def color_status(val):
        if val == "Open":
            return "background-color: rgba(0,200,0,0.1)"
        elif val == "Pending":
            return "background-color: rgba(200,200,0,0.1)"
        elif val == "Partial Fill":
            return "background-color: rgba(0,0,200,0.1)"
        return ""
    
    # Style the dataframe
    styled_df = display_df.style.applymap(
        color_status, 
        subset=["Status"]
    )
    
    # Display the table with orders
    st.dataframe(styled_df, use_container_width=True)
    
    # Order management section
    st.subheader("Order Management")
    
    # Order selection for detailed view
    selected_orders = filtered_df["symbol"].tolist()
    if selected_orders:
        selected_symbol = st.selectbox(
            "Select an order to manage",
            [""] + selected_orders
        )
        
        # Show order details and controls if an order is selected
        if selected_symbol:
            selected_order = filtered_df[filtered_df["symbol"] == selected_symbol].iloc[0]
            
            # Order details
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px;">
                <h3>{selected_symbol} {selected_order['order_type'].replace('_', ' ').title()} Order</h3>
                <p><strong>Side:</strong> {selected_order['side'].upper()}</p>
                <p><strong>Quantity:</strong> {selected_order['quantity']}</p>
                <p><strong>Status:</strong> {selected_order['status'].replace('_', ' ').title()}</p>
                <p><strong>Current Price:</strong> ${selected_order['current_price']:.2f}</p>
                <p><strong>Limit Price:</strong> ${selected_order['limit_price'] if pd.notnull(selected_order['limit_price']) else 'N/A'}</p>
                <p><strong>Stop Price:</strong> ${selected_order['stop_price'] if pd.notnull(selected_order['stop_price']) else 'N/A'}</p>
                <p><strong>Submitted:</strong> {format_time_active(selected_order['minutes_active'])} ago</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Order management controls
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                if st.button(f"Cancel Order", key=f"cancel_{selected_order['order_id']}"):
                    st.warning(f"This would cancel the {selected_symbol} order in a real implementation")
            
            with col_y:
                if st.button(f"Modify Price", key=f"modify_price_{selected_order['order_id']}"):
                    st.info(f"This would modify the price for the {selected_symbol} order in a real implementation")
            
            with col_z:
                if st.button(f"Modify Quantity", key=f"modify_qty_{selected_order['order_id']}"):
                    st.info(f"This would modify the quantity for the {selected_symbol} order in a real implementation")

def render_position_details(position, position_history):
    """
    Render detailed view for a selected position
    """
    # Position overview
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Side indicator (long/short)
    side = position.get("side", "unknown")
    side_color = "green" if side == "long" else "red" if side == "short" else "gray"
    
    # Unrealized P&L formatting
    pnl = position.get("unrealized_pnl", 0)
    pnl_pct = position.get("unrealized_pnl_pct", 0)
    pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "gray"
    
    # Risk metrics
    risk_reward = position.get("risk_reward", 0)
    
    with col1:
        # Position title and basic info
        st.markdown(f"""
        <h2 style="margin-bottom: 0;">{position['symbol']}</h2>
        <p style="font-size: 1.2rem; margin-top: 0; color: {side_color}; font-weight: bold;">
            {side.upper()} {position.get('quantity', '')} @ {position.get('entry_price', '')}
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        # P&L display
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 0;">Unrealized P&L</p>
            <p style="font-size: 1.5rem; font-weight: bold; color: {pnl_color}; margin-top: 0; margin-bottom: 0;">
                ${pnl:,.2f}
            </p>
            <p style="font-size: 1.1rem; color: {pnl_color}; margin-top: 0;">
                {pnl_pct:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Risk/Reward ratio
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 0;">Risk/Reward</p>
            <p style="font-size: 1.5rem; font-weight: bold; margin-top: 0; margin-bottom: 0;">
                1:{risk_reward:.1f}
            </p>
            <p style="font-size: 1.1rem; margin-top: 0;">
                R-Multiple
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Position performance chart
    st.subheader("Position Performance")
    
    if position_history is not None and not position_history.empty:
        # Create the performance chart with stop loss and take profit lines
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=position_history["timestamp"],
                y=position_history["price"],
                mode="lines",
                name="Price",
                line=dict(width=2, color="#1f77b4")
            )
        )
        
        # Add entry price line
        fig.add_trace(
            go.Scatter(
                x=[position_history["timestamp"].min(), position_history["timestamp"].max()],
                y=[position.get("entry_price", 0), position.get("entry_price", 0)],
                mode="lines",
                name="Entry Price",
                line=dict(width=1, color="gray", dash="dash")
            )
        )
        
        # Add stop loss line
        stop_loss = position.get("stop_loss")
        if stop_loss is not None:
            fig.add_trace(
                go.Scatter(
                    x=[position_history["timestamp"].min(), position_history["timestamp"].max()],
                    y=[stop_loss, stop_loss],
                    mode="lines",
                    name="Stop Loss",
                    line=dict(width=1, color="red", dash="dash")
                )
            )
        
        # Add take profit line
        take_profit = position.get("take_profit")
        if take_profit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[position_history["timestamp"].min(), position_history["timestamp"].max()],
                    y=[take_profit, take_profit],
                    mode="lines",
                    name="Take Profit",
                    line=dict(width=1, color="green", dash="dash")
                )
            )
        
        # Enhance the chart layout
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="",
            yaxis_title="Price",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No historical data available for {position['symbol']}")
    
    # Position metrics in a grid
    st.subheader("Position Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Asset Class", position.get("asset_class", "N/A").title())
        st.metric("Days Held", position.get("days_held", "N/A"))
    
    with metrics_col2:
        st.metric("Entry Price", f"${position.get('entry_price', 0):.2f}")
        st.metric("Current Price", f"${position.get('current_price', 0):.2f}")
        
    with metrics_col3:
        st.metric("Stop Loss", f"${position.get('stop_loss', 0):.2f}")
        st.metric("Take Profit", f"${position.get('take_profit', 0):.2f}")
        
    with metrics_col4:
        st.metric("Position Value", f"${position.get('position_value', 0):,.2f}")
        st.metric("Potential Loss", f"${abs(position.get('potential_loss', 0)):,.2f}")
    
    # Position adjustments section
    st.subheader("Position Adjustments")
    
    # Create stop loss adjustment slider
    current_price = position.get("current_price", 0)
    entry_price = position.get("entry_price", 0)
    side = position.get("side", "long")
    
    # Determine appropriate stop loss range based on side
    if side == "long":
        min_stop = max(0.1, current_price * 0.7)  # Don't allow zero or negative stops
        max_stop = current_price * 0.99
        default_stop = position.get("stop_loss", entry_price * 0.9)
    else:  # short
        min_stop = current_price * 1.01
        max_stop = current_price * 1.3
        default_stop = position.get("stop_loss", entry_price * 1.1)
    
    # Stop loss slider
    new_stop_loss = st.slider(
        "Adjust Stop Loss",
        min_value=float(min_stop),
        max_value=float(max_stop),
        value=float(default_stop),
        format="$%.2f"
    )
    
    # Calculate new risk based on adjusted stop
    risk_per_share = abs(entry_price - new_stop_loss)
    potential_loss = risk_per_share * position.get("quantity", 0)
    
    # Display new risk metrics
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-top: 10px;">
        <p style="margin: 0;"><strong>New Risk Per Share:</strong> ${risk_per_share:.2f}</p>
        <p style="margin: 0;"><strong>New Potential Loss:</strong> ${potential_loss:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply stop loss button
    if st.button("Apply New Stop Loss", key=f"apply_stop_{position.get('symbol', 'unknown')}"):
        st.success(f"This would update the stop loss to ${new_stop_loss:.2f} in a real implementation")

def render_trade_blotter(executions_df):
    """
    Render trade execution history (blotter)
    """
    st.subheader("Trade Blotter")
    
    if executions_df.empty:
        st.info("No trade executions found")
        return
    
    # Prepare display DataFrame
    display_cols = [
        "symbol", "action", "quantity", "price", "value",
        "timestamp", "venue", "account_type"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in executions_df.columns:
            executions_df[col] = "N/A"
    
    display_df = executions_df[display_cols].copy()
    
    # Format columns
    display_df["action"] = display_df["action"].str.upper().str.replace("_", " ")
    display_df["value"] = display_df["value"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
    display_df["account_type"] = display_df["account_type"].str.title()
    
    # Format timestamp
    def format_timestamp(ts):
        if isinstance(ts, (datetime.datetime, pd.Timestamp)):
            return ts.strftime("%Y-%m-%d %H:%M")
        return ts
    
    display_df["timestamp"] = display_df["timestamp"].apply(format_timestamp)
    
    # Rename columns for display
    display_df.columns = [
        "Symbol", "Action", "Quantity", "Price", "Value",
        "Time", "Venue", "Account"
    ]
    
    # Add color coding for action
    def color_action(val):
        if "BUY" in val:
            return "color: green"
        elif "SELL" in val:
            return "color: red"
        return ""
    
    # Style the dataframe
    styled_df = display_df.style.applymap(
        color_action, 
        subset=["Action"]
    )
    
    # Display the blotter
    st.dataframe(styled_df, use_container_width=True)

# Main render function for the Orders & Positions component
def render(db, account_type="All Accounts"):
    """
    Main render function for the Orders & Positions section
    """
    # Import mock data generators 
    from dashboard_components.orders_positions import (
        get_positions, get_open_orders, get_position_history, get_execution_history,
        generate_mock_positions_data, generate_mock_orders_data, 
        generate_mock_position_history, generate_mock_executions_data
    )
    
    # Map selected account type to the data filter
    account_type_map = {
        "All Accounts": None,
        "Paper Trading": "paper",
        "Live Trading": "live",
        "Backtest": "backtest"
    }
    
    data_filter = account_type_map.get(account_type)
    
    # Create tabs for positions, orders, and blotter
    tab1, tab2, tab3 = st.tabs(["Positions", "Orders", "Trade Blotter"])
    
    with tab1:
        # Get positions data
        positions_df = get_positions(db, data_filter)
        
        # Render positions table and management interface
        render_positions_table(positions_df, account_type)
    
    with tab2:
        # Get open orders data
        orders_df = get_open_orders(db, data_filter)
        
        # Render orders table and management interface
        render_open_orders_table(orders_df, account_type)
    
    with tab3:
        # Get execution history data
        executions_df = get_execution_history(db, data_filter)
        
        # Render trade blotter
        render_trade_blotter(executions_df)
