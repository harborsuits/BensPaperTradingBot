"""
Signals & Approvals UI Rendering Functions for BensBot Dashboard
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

def render_trading_signals(signals_df, account_type):
    """
    Render trading signals with filtering and management controls
    """
    st.subheader(f"Trading Signals - {account_type}")
    
    if signals_df.empty:
        st.info(f"No trading signals found for {account_type}")
        return
    
    # Create upcoming/past signal tabs
    tab1, tab2 = st.tabs(["Upcoming Signals", "Signal History"])
    
    with tab1:
        # Filter for upcoming signals (status = pending)
        upcoming_df = signals_df[signals_df["status"] == "pending"].copy()
        
        if upcoming_df.empty:
            st.info("No upcoming signals found")
        else:
            render_upcoming_signals(upcoming_df)
    
    with tab2:
        # Filter for past signals (status != pending)
        past_df = signals_df[signals_df["status"] != "pending"].copy()
        
        if past_df.empty:
            st.info("No signal history found")
        else:
            render_signal_history_table(past_df)

def render_upcoming_signals(signals_df):
    """
    Render upcoming signals with approval controls
    """
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Upcoming Signals", f"{len(signals_df)}")
    
    with col2:
        # Count by direction
        long_count = len(signals_df[signals_df["direction"] == "long"])
        short_count = len(signals_df[signals_df["direction"] == "short"])
        dominant_direction = "Long" if long_count >= short_count else "Short"
        direction_count = long_count if long_count >= short_count else short_count
        st.metric("Signal Bias", f"{dominant_direction}: {direction_count}")
    
    with col3:
        # Average confidence
        avg_confidence = signals_df["confidence"].mean() if "confidence" in signals_df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Signal filtering options
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Filter by signal type
        if "signal_type" in signals_df.columns:
            signal_types = ["All Types"] + sorted(signals_df["signal_type"].unique().tolist())
            signal_type_filter = st.selectbox(
                "Signal Type",
                options=signal_types
            )
        else:
            signal_type_filter = "All Types"
    
    with col_b:
        # Filter by direction
        if "direction" in signals_df.columns:
            direction_filter = st.radio(
                "Direction",
                ["All", "Long Only", "Short Only"],
                horizontal=True
            )
        else:
            direction_filter = "All"
    
    # Apply filters
    filtered_df = signals_df.copy()
    
    if signal_type_filter != "All Types" and "signal_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["signal_type"] == signal_type_filter]
    
    if direction_filter == "Long Only" and "direction" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["direction"] == "long"]
    elif direction_filter == "Short Only" and "direction" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["direction"] == "short"]
    
    # Prepare display DataFrame
    display_cols = [
        "symbol", "strategy_name", "signal_type", "direction", "signal_price",
        "timestamp", "strength", "confidence"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = "N/A"
    
    display_df = filtered_df[display_cols].copy()
    
    # Format columns
    display_df["direction"] = display_df["direction"].str.upper()
    display_df["signal_type"] = display_df["signal_type"].str.replace("_", " ").str.title()
    
    # Format timestamp as time until execution
    def format_time_until(ts):
        if not isinstance(ts, (datetime.datetime, pd.Timestamp)):
            return ts
        
        now = datetime.datetime.now()
        if ts < now:
            return "Due now"
        
        delta = ts - now
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if delta.days > 0:
            return f"In {delta.days}d {hours}h"
        elif hours > 0:
            return f"In {hours}h {minutes}m"
        else:
            return f"In {minutes}m"
    
    display_df["timestamp"] = display_df["timestamp"].apply(format_time_until)
    
    # Format confidence as percentage
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0%}" if isinstance(x, (int, float)) else x)
    
    # Rename columns for display
    display_df.columns = [
        "Symbol", "Strategy", "Type", "Direction", "Signal Price",
        "Executes", "Strength", "Confidence"
    ]
    
    # Add color coding for direction
    def color_direction(val):
        if val == "LONG":
            return "color: green"
        elif val == "SHORT":
            return "color: red"
        return ""
    
    # Style the dataframe
    styled_df = display_df.style.applymap(
        color_direction, 
        subset=["Direction"]
    )
    
    # Display the upcoming signals table
    st.dataframe(styled_df, use_container_width=True)
    
    # Signal selection for detailed view
    st.subheader("Signal Details")
    
    # Get the signal IDs from the original dataframe
    signal_ids = filtered_df["signal_id"].tolist() if "signal_id" in filtered_df.columns else []
    
    if signal_ids:
        # Map display names to signal IDs
        signal_options = {f"{row['symbol']} - {row['strategy_name']} ({row['signal_type']})": row["signal_id"] 
                         for _, row in filtered_df.iterrows()}
        
        selected_signal_display = st.selectbox(
            "Select a signal to view details",
            [""] + list(signal_options.keys())
        )
        
        # Show signal details if a signal is selected
        if selected_signal_display:
            selected_signal_id = signal_options[selected_signal_display]
            selected_signal = filtered_df[filtered_df["signal_id"] == selected_signal_id].iloc[0]
            
            render_signal_details(selected_signal)

def render_signal_history_table(signals_df):
    """
    Render table of historical signals with filtering
    """
    # Filter options
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        # Filter by strategy
        if "strategy_name" in signals_df.columns:
            strategy_options = ["All Strategies"] + sorted(signals_df["strategy_name"].unique().tolist())
            strategy_filter = st.selectbox(
                "Strategy",
                options=strategy_options
            )
        else:
            strategy_filter = "All Strategies"
    
    with col_b:
        # Filter by status
        if "status" in signals_df.columns:
            status_options = ["All Statuses"] + sorted(signals_df["status"].unique().tolist())
            status_filter = st.selectbox(
                "Status",
                options=status_options
            )
        else:
            status_filter = "All Statuses"
    
    with col_c:
        # Filter by time period
        time_options = ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
        time_filter = st.selectbox(
            "Time Period",
            options=time_options
        )
    
    # Apply filters
    filtered_df = signals_df.copy()
    
    if strategy_filter != "All Strategies" and "strategy_name" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["strategy_name"] == strategy_filter]
    
    if status_filter != "All Statuses" and "status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["status"] == status_filter]
    
    # Apply time filter
    if "timestamp" in filtered_df.columns:
        now = datetime.datetime.now()
        if time_filter == "Last 24 Hours":
            cutoff = now - datetime.timedelta(days=1)
            filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]
        elif time_filter == "Last 7 Days":
            cutoff = now - datetime.timedelta(days=7)
            filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]
        elif time_filter == "Last 30 Days":
            cutoff = now - datetime.timedelta(days=30)
            filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]
    
    # Prepare display DataFrame
    display_cols = [
        "symbol", "strategy_name", "signal_type", "direction", "signal_price",
        "timestamp", "status", "pnl_pct"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in filtered_df.columns:
            filtered_df[col] = "N/A"
    
    display_df = filtered_df[display_cols].copy()
    
    # Format columns
    display_df["direction"] = display_df["direction"].str.upper()
    display_df["signal_type"] = display_df["signal_type"].str.replace("_", " ").str.title()
    display_df["status"] = display_df["status"].str.replace("_", " ").str.title()
    
    # Format timestamp
    def format_timestamp(ts):
        if not isinstance(ts, (datetime.datetime, pd.Timestamp)):
            return ts
        
        now = datetime.datetime.now()
        delta = now - ts
        
        if delta.days > 7:
            return ts.strftime("%Y-%m-%d")
        elif delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h ago"
        else:
            return f"{delta.seconds // 60}m ago"
    
    display_df["timestamp"] = display_df["timestamp"].apply(format_timestamp)
    
    # Format P&L
    if "pnl_pct" in display_df.columns:
        display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A")
    
    # Rename columns for display
    display_df.columns = [
        "Symbol", "Strategy", "Type", "Direction", "Signal Price",
        "Time", "Status", "P&L %"
    ]
    
    # Add color coding for P&L
    def color_pnl(val):
        if isinstance(val, str) and "%" in val:
            try:
                num_val = float(val.replace("%", ""))
                if num_val > 0:
                    return "color: green"
                elif num_val < 0:
                    return "color: red"
            except:
                pass
        return ""
    
    # Style the dataframe
    styled_df = display_df.style.applymap(
        color_pnl, 
        subset=["P&L %"]
    )
    
    # Display the signal history table
    st.dataframe(styled_df, use_container_width=True)
    
    # Show performance metrics if there are executed signals
    executed_df = filtered_df[filtered_df["status"].isin(["executed", "triggered"])]
    if not executed_df.empty and "is_winner" in executed_df.columns:
        st.subheader("Signal Performance")
        
        # Calculate performance metrics
        wins = executed_df["is_winner"].sum()
        total = len(executed_df)
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        avg_win = executed_df.loc[executed_df["is_winner"] == True, "pnl_pct"].mean() if "pnl_pct" in executed_df.columns else 0
        avg_loss = executed_df.loc[executed_df["is_winner"] == False, "pnl_pct"].mean() if "pnl_pct" in executed_df.columns else 0
        
        # Display metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with metrics_col2:
            st.metric("Signals", f"{total}")
        
        with metrics_col3:
            st.metric("Avg Win", f"{avg_win:.2f}%" if avg_win else "N/A")
        
        with metrics_col4:
            st.metric("Avg Loss", f"{avg_loss:.2f}%" if avg_loss else "N/A")

def render_signal_details(signal):
    """
    Render detailed view for a selected signal
    """
    # Signal overview
    col1, col2 = st.columns([2, 1])
    
    # Direction indicator (long/short)
    direction = signal.get("direction", "unknown")
    direction_color = "green" if direction == "long" else "red" if direction == "short" else "gray"
    
    with col1:
        # Signal title and basic info
        st.markdown(f"""
        <h3 style="margin-bottom: 0;">{signal['symbol']} - {signal['signal_type'].replace('_', ' ').title()}</h3>
        <p style="font-size: 1.2rem; margin-top: 0; color: {direction_color}; font-weight: bold;">
            {direction.upper()} Signal from {signal.get('strategy_name', 'Unknown Strategy')}
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        # Signal strength/confidence
        strength = signal.get("strength", "moderate").title()
        confidence = signal.get("confidence", 0.5)
        
        # Map strength to color
        strength_color = {
            "Strong": "green",
            "Moderate": "blue",
            "Weak": "orange"
        }.get(strength, "gray")
        
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 0;">Signal Strength</p>
            <p style="font-size: 1.3rem; font-weight: bold; color: {strength_color}; margin-top: 0; margin-bottom: 0;">
                {strength}
            </p>
            <p style="font-size: 1.1rem; margin-top: 0;">
                {confidence:.0%} Confidence
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Price information
    st.subheader("Price Information")
    
    price_col1, price_col2, price_col3 = st.columns(3)
    
    with price_col1:
        # Signal price
        signal_price = signal.get("signal_price")
        if signal_price is not None:
            st.metric("Signal Price", f"${signal_price:.2f}")
        
    with price_col2:
        # Stop price
        stop_price = signal.get("stop_price")
        if stop_price is not None:
            st.metric("Stop Loss", f"${stop_price:.2f}")
    
    with price_col3:
        # Target price
        target_price = signal.get("target_price")
        if target_price is not None:
            st.metric("Take Profit", f"${target_price:.2f}")
    
    # Risk/reward visualization
    if signal.get("signal_price") is not None and signal.get("stop_price") is not None and signal.get("target_price") is not None:
        # Create risk/reward visualization
        st.subheader("Risk/Reward Visualization")
        
        # Calculate distances
        risk = abs(signal["signal_price"] - signal["stop_price"])
        reward = abs(signal["target_price"] - signal["signal_price"])
        risk_reward_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        # Create the chart
        fig = go.Figure()
        
        # Calculate appropriate y-range
        min_price = min(signal["signal_price"], signal["stop_price"], signal["target_price"]) * 0.95
        max_price = max(signal["signal_price"], signal["stop_price"], signal["target_price"]) * 1.05
        
        # Add price levels
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[signal["signal_price"], signal["signal_price"]],
                mode="lines+text",
                name="Signal Price",
                line=dict(color="blue", width=2),
                text=["", f"Signal: ${signal['signal_price']:.2f}"],
                textposition="middle right"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[signal["stop_price"], signal["stop_price"]],
                mode="lines+text",
                name="Stop Loss",
                line=dict(color="red", width=2),
                text=["", f"Stop: ${signal['stop_price']:.2f}"],
                textposition="middle right"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[signal["target_price"], signal["target_price"]],
                mode="lines+text",
                name="Take Profit",
                line=dict(color="green", width=2),
                text=["", f"Target: ${signal['target_price']:.2f}"],
                textposition="middle right"
            )
        )
        
        # Add risk/reward annotation
        fig.add_annotation(
            x=0.5, y=min_price,
            text=f"Risk/Reward: 1:{risk_reward_ratio}",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Update layout
        fig.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=50),
            yaxis=dict(range=[min_price, max_price]),
            xaxis=dict(showticklabels=False, range=[-0.1, 1.2])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Signal details
    st.subheader("Signal Details")
    
    # Format timestamp
    timestamp = signal.get("timestamp")
    if isinstance(timestamp, (datetime.datetime, pd.Timestamp)):
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
    else:
        formatted_time = str(timestamp)
    
    # Create details section
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px;">
        <p><strong>Signal ID:</strong> {signal.get('signal_id', 'N/A')}</p>
        <p><strong>Strategy:</strong> {signal.get('strategy_name', 'N/A')}</p>
        <p><strong>Asset Class:</strong> {signal.get('asset_class', 'N/A').title()}</p>
        <p><strong>Timestamp:</strong> {formatted_time}</p>
        <p><strong>Quantity:</strong> {signal.get('quantity', 'N/A')}</p>
        <p><strong>Status:</strong> {signal.get('status', 'N/A').replace('_', ' ').title()}</p>
        <p><strong>Notes:</strong> {signal.get('notes', 'No additional notes.')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Signal management controls
    st.subheader("Signal Actions")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        if st.button("Approve Signal", key=f"approve_{signal.get('signal_id', 'unknown')}"):
            st.success("This would approve the signal for execution in a real implementation")
    
    with col_y:
        if st.button("Reject Signal", key=f"reject_{signal.get('signal_id', 'unknown')}"):
            st.error("This would reject the signal in a real implementation")

def render_pending_approvals(approvals_df, account_type):
    """
    Render strategies pending approval with approval interface
    """
    st.subheader(f"Pending Strategy Approvals - {account_type}")
    
    if approvals_df.empty:
        st.info(f"No strategies pending approval for {account_type}")
        return
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Pending Approvals", f"{len(approvals_df)}")
    
    with col2:
        if "request_date" in approvals_df.columns:
            oldest_request = approvals_df["request_date"].min()
            st.metric("Oldest Request", f"{oldest_request}")
    
    # Strategy filtering options
    if "category" in approvals_df.columns:
        categories = ["All Categories"] + sorted(approvals_df["category"].unique().tolist())
        category_filter = st.selectbox(
            "Strategy Category",
            options=categories
        )
        
        # Apply filter
        if category_filter != "All Categories":
            approvals_df = approvals_df[approvals_df["category"] == category_filter]
    
    # Prepare display DataFrame
    display_cols = [
        "name", "category", "win_rate", "profit_factor", 
        "sharpe_ratio", "max_drawdown", "request_date"
    ]
    
    # Ensure all required columns exist
    for col in display_cols:
        if col not in approvals_df.columns:
            approvals_df[col] = "N/A"
    
    display_df = approvals_df[display_cols].copy()
    
    # Format columns
    display_df["category"] = display_df["category"].str.replace("_", " ").str.title()
    display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    # Rename columns for display
    display_df.columns = [
        "Strategy Name", "Category", "Win Rate", "Profit Factor",
        "Sharpe Ratio", "Max Drawdown", "Request Date"
    ]
    
    # Display the pending approvals table
    st.dataframe(display_df, use_container_width=True)
    
    # Strategy selection for detailed view
    st.subheader("Strategy Approval")
    
    selected_strategy_name = st.selectbox(
        "Select a strategy to review",
        [""] + approvals_df["name"].tolist()
    )
    
    # Show strategy details and approval controls if a strategy is selected
    if selected_strategy_name:
        selected_strategy = approvals_df[approvals_df["name"] == selected_strategy_name].iloc[0]
        
        render_strategy_approval(selected_strategy)

def render_strategy_approval(strategy):
    """
    Render detailed view for a strategy pending approval
    """
    # Strategy overview
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Strategy title and basic info
        st.markdown(f"""
        <h3 style="margin-bottom: 0;">{strategy['name']}</h3>
        <p style="font-size: 1.1rem; margin-top: 0; color: #666;">
            {strategy.get('category', 'Unknown Category').replace('_', ' ').title()} Strategy
        </p>
        <p>{strategy.get('description', 'No description available.')}</p>
        """, unsafe_allow_html=True)
    
    with col2:
        # Request details
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; text-align: center;">
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 0;">Approval Request Date</p>
            <p style="font-size: 1.2rem; font-weight: bold; margin-top: 0;">
                {strategy.get('request_date', 'Unknown')}
            </p>
            <p style="font-size: 0.9rem; color: #666; margin-top: 10px; margin-bottom: 0;">Status</p>
            <p style="font-size: 1.2rem; font-weight: bold; color: blue; margin-top: 0;">
                Pending Approval
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Win Rate", f"{strategy.get('win_rate', 0):.1f}%")
        st.metric("Total Trades", f"{strategy.get('total_trades', 0)}")
    
    with metrics_col2:
        st.metric("Profit Factor", f"{strategy.get('profit_factor', 0):.2f}")
        st.metric("Annual Return", f"{strategy.get('annual_return', 0):.1f}%")
        
    with metrics_col3:
        st.metric("Sharpe Ratio", f"{strategy.get('sharpe_ratio', 0):.2f}")
        st.metric("Max Drawdown", f"{strategy.get('max_drawdown', 0):.1f}%")
        
    with metrics_col4:
        # Additional metrics
        avg_win = strategy.get('avg_win', 0)
        avg_loss = strategy.get('avg_loss', 0)
        if avg_win and avg_loss:
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}")
        
        st.metric("Expected Value", f"{strategy.get('win_rate', 0)/100 * strategy.get('profit_factor', 0):.2f}")
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    # Get parameters if available
    params = strategy.get('parameters', {})
    
    if params and isinstance(params, dict):
        # Create parameter table
        param_data = []
        for param_name, param_value in params.items():
            param_data.append({
                "Parameter": param_name.replace("_", " ").title(),
                "Value": param_value
            })
        
        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True)
    else:
        st.info("No parameter information available.")
    
    # Approval controls
    st.subheader("Approval Decision")
    
    # Notes input
    approval_notes = st.text_area(
        "Approval Notes",
        placeholder="Enter any notes regarding your approval decision..."
    )
    
    col_x, col_y, col_z = st.columns(3)
    
    with col_x:
        if st.button("Approve Strategy", key=f"approve_{strategy.get('strategy_id', 'unknown')}"):
            st.success("This would approve the strategy in a real implementation")
    
    with col_y:
        if st.button("Reject Strategy", key=f"reject_{strategy.get('strategy_id', 'unknown')}"):
            st.error("This would reject the strategy in a real implementation")
    
    with col_z:
        if st.button("Request Changes", key=f"request_changes_{strategy.get('strategy_id', 'unknown')}"):
            st.warning("This would request changes to the strategy in a real implementation")

# Main render function for the Signals & Approvals component
def render(db, account_type="All Accounts"):
    """
    Main render function for the Signals & Approvals section
    """
    # Import data retrieval functions
    from dashboard_components.signals_approvals import (
        get_trading_signals, get_pending_approvals, get_signal_history,
        generate_mock_signals_data, generate_mock_approvals_data, generate_mock_signal_history
    )
    
    # Map selected account type to the data filter
    account_type_map = {
        "All Accounts": None,
        "Paper Trading": "paper",
        "Live Trading": "live",
        "Backtest": "backtest"
    }
    
    data_filter = account_type_map.get(account_type)
    
    # Create tabs for signals and approvals
    tab1, tab2 = st.tabs(["Trading Signals", "Strategy Approvals"])
    
    with tab1:
        # Get trading signals data
        signals_df = get_trading_signals(db, data_filter)
        
        # Render trading signals interface
        render_trading_signals(signals_df, account_type)
    
    with tab2:
        # Get pending approvals data
        approvals_df = get_pending_approvals(db, data_filter)
        
        # Render pending approvals interface
        render_pending_approvals(approvals_df, account_type)
