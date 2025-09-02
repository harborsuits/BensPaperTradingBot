"""
Strategy Library Component for BensBot Dashboard
Displays strategy performance across account types with approval controls
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Get strategy data from MongoDB
def get_strategy_data(db, account_type=None):
    """
    Retrieve strategy data from MongoDB for the specified account type
    If account_type is None, return data for all account types
    """
    if db is None:
        return generate_mock_strategy_data(account_type)
    
    try:
        # Query filter based on account_type
        query = {}
        if account_type and account_type != "All":
            query["account_type"] = account_type.lower().replace(" trading", "").strip()
        
        # Get strategy documents from MongoDB
        strategy_docs = list(db.strategies.find(query))
        
        if strategy_docs:
            # Convert to DataFrame
            df = pd.DataFrame(strategy_docs)
            return df
        else:
            # If no data found, generate mock data
            return generate_mock_strategy_data(account_type)
    except Exception as e:
        st.error(f"Error retrieving strategy data: {e}")
        return generate_mock_strategy_data(account_type)

# Generate mock strategy data for development/testing
def generate_mock_strategy_data(account_type=None):
    """Generate synthetic strategy data for development and testing"""
    # Strategy templates with consistent IDs across account types
    strategy_templates = [
        {"id": "s001", "name": "TrendFollower", "category": "momentum", 
         "description": "Classic trend following strategy using SMA crossovers"},
        {"id": "s002", "name": "BreakoutHunter", "category": "breakout", 
         "description": "Detects and trades price breakouts with ATR-based risk management"},
        {"id": "s003", "name": "MeanReverter", "category": "mean_reversion", 
         "description": "Mean reversion using Bollinger Bands and RSI"},
        {"id": "s004", "name": "VolatilityHarvester", "category": "volatility", 
         "description": "Options-based volatility harvesting strategy"},
        {"id": "s005", "name": "MacroEdge", "category": "macro", 
         "description": "Macro factors-driven strategy using economic indicators"},
        {"id": "s006", "name": "SectorRotator", "category": "rotation", 
         "description": "Sector rotation based on market cycle analysis"},
        {"id": "s007", "name": "DividendHarvester", "category": "income", 
         "description": "Focuses on stable dividend income with low volatility"},
        {"id": "s008", "name": "SwingTrader", "category": "swing", 
         "description": "Multi-day swing trading using momentum and sentiment"},
        {"id": "s009", "name": "PatternRecognizer", "category": "pattern", 
         "description": "Trades chart patterns with machine learning validation"},
        {"id": "s010", "name": "TechnicalEnsemble", "category": "ensemble", 
         "description": "Ensemble of technical indicators with adaptive weighting"}
    ]
    
    # Account type specific performance characteristics 
    account_type_chars = {
        "live": {"win_rate_range": (52, 68), "trades_range": (10, 50), 
                "return_range": (5, 15), "drawdown_range": (-2, -8),
                "sharpe_range": (1.0, 1.8), "approval_prob": 0.7},
        
        "paper": {"win_rate_range": (48, 72), "trades_range": (20, 100), 
                "return_range": (8, 25), "drawdown_range": (-4, -12),
                "sharpe_range": (0.8, 2.2), "approval_prob": 0.8},
        
        "backtest": {"win_rate_range": (55, 75), "trades_range": (100, 500), 
                    "return_range": (15, 35), "drawdown_range": (-5, -15),
                    "sharpe_range": (1.2, 2.8), "approval_prob": 0.9}
    }
    
    # Map account_type parameter to our internal keys
    account_type_key = None
    if account_type:
        account_type = account_type.lower().replace(" trading", "").strip()
        if account_type in account_type_chars:
            account_type_key = account_type
    
    # If no specific account type requested, or invalid type, use all
    if account_type_key is None:
        account_types_to_generate = list(account_type_chars.keys())
    else:
        account_types_to_generate = [account_type_key]
    
    # Strategy statuses and their probabilities
    statuses = ["active", "inactive", "pending_approval", "rejected"]
    status_probs = {"active": 0.5, "inactive": 0.2, "pending_approval": 0.2, "rejected": 0.1}
    
    # Generate strategies for each account type
    all_strategies = []
    
    for act in account_types_to_generate:
        chars = account_type_chars[act]
        
        for template in strategy_templates:
            # Copy the template
            strategy = template.copy()
            
            # Add account-type specific fields
            strategy["account_type"] = act
            strategy["strategy_id"] = f"{template['id']}_{act}"
            
            # Generate random performance metrics based on account type characteristics
            strategy["win_rate"] = round(random.uniform(*chars["win_rate_range"]), 1)
            strategy["total_trades"] = random.randint(*chars["trades_range"])
            strategy["annual_return"] = round(random.uniform(*chars["return_range"]), 1)
            strategy["max_drawdown"] = round(random.uniform(*chars["drawdown_range"]), 1)
            strategy["sharpe_ratio"] = round(random.uniform(*chars["sharpe_range"]), 2)
            
            # Calculate profit factor (roughly based on win rate)
            win_rate_decimal = strategy["win_rate"] / 100
            avg_win = random.uniform(1.5, 2.5)
            avg_loss = 1.0  # Normalized to 1.0
            strategy["profit_factor"] = round((win_rate_decimal * avg_win) / 
                                            ((1 - win_rate_decimal) * avg_loss), 2)
            
            # Calculate other metrics
            strategy["avg_holding_days"] = round(random.uniform(1.5, 12.0), 1)
            strategy["volatility"] = round(random.uniform(5.0, 25.0), 1)
            
            # Status based on probabilistic assignment
            rand_val = random.random()
            cumulative = 0
            for status, prob in status_probs.items():
                cumulative += prob
                if rand_val <= cumulative:
                    strategy["status"] = status
                    break
            
            # Approval flag based on account type (more stringent for live)
            strategy["approved"] = random.random() < chars["approval_prob"]
            
            # Last updated timestamp
            days_ago = random.randint(1, 30)
            strategy["last_updated"] = (datetime.datetime.now() - 
                                        datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Add to result
            all_strategies.append(strategy)
    
    # Convert to DataFrame
    return pd.DataFrame(all_strategies)

# Format strategy data for display
def format_strategy_data(df):
    """Format strategy DataFrame for display in the UI"""
    # Select and order columns for display
    display_cols = [
        "name", "category", "win_rate", "profit_factor", 
        "annual_return", "max_drawdown", "sharpe_ratio", 
        "total_trades", "status", "approved"
    ]
    
    # Ensure all needed columns exist
    for col in display_cols:
        if col not in df.columns:
            if col in ["win_rate", "profit_factor", "annual_return", 
                      "max_drawdown", "sharpe_ratio", "total_trades"]:
                df[col] = 0
            else:
                df[col] = ""
    
    # Create a copy with only the display columns
    display_df = df[display_cols].copy()
    
    # Format columns
    display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    display_df["annual_return"] = display_df["annual_return"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    display_df["max_drawdown"] = display_df["max_drawdown"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    # Rename columns for display
    display_df.columns = [
        "Strategy", "Category", "Win Rate", "Profit Factor", 
        "Annual Return", "Max Drawdown", "Sharpe", 
        "Trades", "Status", "Approved"
    ]
    
    return display_df

# Get strategy performance history
def get_strategy_performance_history(db, strategy_id, account_type):
    """Get historical performance data for a specific strategy"""
    if db is None:
        return generate_mock_strategy_history(strategy_id, account_type)
    
    try:
        # Query MongoDB for strategy performance history
        history_docs = list(db.strategy_performance.find({
            "strategy_id": strategy_id,
            "account_type": account_type
        }).sort("date", 1))
        
        if history_docs:
            df = pd.DataFrame(history_docs)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return generate_mock_strategy_history(strategy_id, account_type)
    except Exception as e:
        st.error(f"Error retrieving strategy performance history: {e}")
        return generate_mock_strategy_history(strategy_id, account_type)

# Generate mock strategy performance history
def generate_mock_strategy_history(strategy_id, account_type):
    """Generate synthetic performance history for a strategy"""
    # Extract base strategy_id without account_type suffix if present
    base_strategy_id = strategy_id.split('_')[0] if '_' in strategy_id else strategy_id
    
    # Set seed based on strategy_id for consistent results
    seed_val = sum(ord(c) for c in base_strategy_id)
    random.seed(seed_val)
    
    # Period and frequency
    days = 180  # 6 months history
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(days=i) for i in range(days, 0, -1)]
    
    # Starting equity and characteristics based on account_type
    if account_type == "live":
        starting_equity = 10000
        trend = 0.0005  # 0.05% daily drift
        volatility = 0.006  # 0.6% daily volatility
    elif account_type == "paper":
        starting_equity = 10000
        trend = 0.0008  # 0.08% daily drift
        volatility = 0.009  # 0.9% daily volatility
    else:  # backtest
        starting_equity = 10000
        trend = 0.001  # 0.1% daily drift
        volatility = 0.007  # 0.7% daily volatility
    
    # Generate equity curve
    equity = [starting_equity]
    cumulative_return = [0.0]
    drawdown = [0.0]
    
    peak = starting_equity
    
    for i in range(1, days):
        # Random walk with trend
        change = np.random.normal(trend, volatility) * equity[-1]
        new_value = max(equity[-1] + change, 0.5 * starting_equity)  # Don't go below 50% of starting equity
        
        equity.append(new_value)
        
        # Update peak and calculate drawdown
        peak = max(peak, new_value)
        current_drawdown = (new_value - peak) / peak * 100
        drawdown.append(current_drawdown)
        
        # Calculate cumulative return
        curr_return = (new_value - starting_equity) / starting_equity * 100
        cumulative_return.append(curr_return)
    
    # Create DataFrame
    result = pd.DataFrame({
        'date': dates,
        'equity': equity,
        'cumulative_return': cumulative_return,
        'drawdown': drawdown,
        'strategy_id': strategy_id,
        'account_type': account_type
    })
    
    # Reset random seed
    random.seed()
    
    return result

# Render strategy list with filters
def render_strategy_list(strategies_df, account_type):
    """Render the filterable strategy table with performance metrics"""
    st.subheader(f"Strategy Library - {account_type}")
    
    # Sidebar filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Category filter
        categories = ["All Categories"] + sorted(strategies_df["category"].unique().tolist())
        selected_category = st.selectbox("Category", categories)
    
    with col2:
        # Status filter
        statuses = ["All Statuses"] + sorted(strategies_df["status"].unique().tolist())
        status_display = {
            "active": "Active",
            "inactive": "Inactive",
            "pending_approval": "Pending Approval",
            "rejected": "Rejected"
        }
        status_options = ["All Statuses"] + [status_display.get(s, s) for s in strategies_df["status"].unique()]
        selected_status_display = st.selectbox("Status", status_options)
        # Map display name back to data value
        selected_status = next((k for k, v in status_display.items() if v == selected_status_display), 
                              selected_status_display)
    
    with col3:
        # Performance filter (e.g., minimum Sharpe ratio)
        min_sharpe = st.slider("Min Sharpe Ratio", 0.0, 3.0, 0.0, 0.1)
    
    # Apply filters
    filtered_df = strategies_df.copy()
    
    if selected_category != "All Categories":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    
    if selected_status != "All Statuses":
        filtered_df = filtered_df[filtered_df["status"] == selected_status]
    
    if min_sharpe > 0:
        filtered_df = filtered_df[filtered_df["sharpe_ratio"] >= min_sharpe]
    
    # Prepare for display
    if not filtered_df.empty:
        display_df = format_strategy_data(filtered_df)
        
        # Add color coding to status column
        def color_status(val):
            if val == "Active":
                return "background-color: rgba(0,200,0,0.2)"
            elif val == "Inactive":
                return "background-color: rgba(200,200,0,0.2)"
            elif val == "Pending Approval":
                return "background-color: rgba(0,0,200,0.2)"
            elif val == "Rejected":
                return "background-color: rgba(200,0,0,0.2)"
            return ""
        
        # Style the dataframe
        styled_df = display_df.style.map(lambda x: color_status(x) if x in ["Active", "Inactive", "Pending Approval", "Rejected"] else "", subset=["Status"])
        
        # Display the table with strategy details
        st.dataframe(styled_df, use_container_width=True)
        
        # Strategy selection for detailed view
        st.subheader("Strategy Details")
        selected_strategy_name = st.selectbox(
            "Select a strategy to view details",
            [""] + filtered_df["name"].tolist()
        )
        
        # Show detailed view if a strategy is selected
        if selected_strategy_name:
            selected_strategy = filtered_df[filtered_df["name"] == selected_strategy_name].iloc[0]
            render_strategy_details(selected_strategy, account_type)
    else:
        st.info("No strategies match the selected filters.")

# Render detailed view for a selected strategy
def render_strategy_details(strategy, account_type):
    """Render detailed view for a selected strategy"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Strategy name and description
        st.markdown(f"""
        <h3 style="margin-bottom: 0;">{strategy['name']}</h3>
        <p style="color: #666; margin-top: 0; text-transform: capitalize;">{strategy['category']} Strategy</p>
        <p>{strategy.get('description', 'No description available.')}</p>
        """, unsafe_allow_html=True)
        
        # Performance metrics in a grid
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Win Rate", f"{strategy['win_rate']}%")
            st.metric("Total Trades", f"{strategy['total_trades']}")
        
        with metrics_col2:
            st.metric("Annual Return", f"{strategy['annual_return']}%")
            st.metric("Profit Factor", f"{strategy['profit_factor']}")
            
        with metrics_col3:
            st.metric("Max Drawdown", f"{strategy['max_drawdown']}%")
            st.metric("Avg Holding", f"{strategy.get('avg_holding_days', 'N/A')} days")
            
        with metrics_col4:
            st.metric("Sharpe Ratio", f"{strategy['sharpe_ratio']}")
            st.metric("Volatility", f"{strategy.get('volatility', 'N/A')}%")
    
    with col2:
        # Strategy status and controls
        status = strategy['status']
        approved = strategy.get('approved', False)
        
        # Status indicator
        status_color = {
            "active": "green",
            "inactive": "orange",
            "pending_approval": "blue",
            "rejected": "red"
        }.get(status, "gray")
        
        status_display = {
            "active": "Active",
            "inactive": "Inactive",
            "pending_approval": "Pending Approval",
            "rejected": "Rejected"
        }.get(status, status.title())
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Status</p>
            <p style="font-size: 1.2rem; font-weight: 700; color: {status_color}; margin-bottom: 15px;">{status_display}</p>
            
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Approval</p>
            <p style="font-size: 1.2rem; font-weight: 700; color: {'green' if approved else 'red'}; margin-bottom: 15px;">{'Approved' if approved else 'Not Approved'}</p>
            
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #666;">Last Updated</p>
            <p style="font-size: 1.2rem; margin-bottom: 0;">{strategy.get('last_updated', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Strategy controls
        if status == "active":
            if st.button("Deactivate Strategy", key=f"deactivate_{strategy['id']}"):
                st.warning("This would deactivate the strategy in a real implementation")
        elif status == "inactive":
            if st.button("Activate Strategy", key=f"activate_{strategy['id']}"):
                st.success("This would activate the strategy in a real implementation")
        
        if status == "pending_approval":
            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("Approve", key=f"approve_{strategy['id']}"):
                    st.success("Strategy would be approved in a real implementation")
            with col_reject:
                if st.button("Reject", key=f"reject_{strategy['id']}"):
                    st.error("Strategy would be rejected in a real implementation")
    
    # Performance history chart
    st.subheader("Performance History")
    
    # Get historical data (real or mock)
    history_df = get_strategy_performance_history(None, strategy.get('strategy_id', strategy.get('id')), 
                                               account_type.lower().replace(" trading", "").strip())
    
    # Create figures
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Equity Curve", "Drawdown"),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08
    )
    
    # Add equity curve trace
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(width=2, color="#1f77b4")
        ),
        row=1, col=1
    )
    
    # Add drawdown trace
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df["drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(width=2, color="#d62728"),
            fill="tozeroy"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white",
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade distribution section
    st.subheader("Trade Analytics")
    
    # Create mock trade distribution data
    bins = np.linspace(-5, 5, 20)
    trade_returns = np.random.normal(0.2, 1.5, strategy['total_trades'])
    
    # Trade distribution histogram
    dist_fig = go.Figure()
    
    dist_fig.add_trace(
        go.Histogram(
            x=trade_returns,
            marker_color="#1f77b4",
            opacity=0.7,
            name="Trade Returns (%)"
        )
    )
    
    dist_fig.update_layout(
        title="Trade Return Distribution",
        xaxis_title="Return per Trade (%)",
        yaxis_title="Number of Trades",
        plot_bgcolor="white"
    )
    
    # Update axes
    dist_fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    dist_fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    
    st.plotly_chart(dist_fig, use_container_width=True)

# Main render function for this component
def render(db, account_type="All Accounts"):
    """Main render function for the Strategy Library section"""
    
    # Map selected account type to the data filter
    account_type_map = {
        "All Accounts": None,
        "Paper Trading": "paper",
        "Live Trading": "live",
        "Backtest": "backtest"
    }
    
    data_filter = account_type_map.get(account_type)
    
    # Get strategies data
    strategies_df = get_strategy_data(db, data_filter)
    
    # Create tabs for different account types
    if data_filter is None:
        # If no specific account type selected, show tabs
        tab1, tab2, tab3 = st.tabs(["Live Trading", "Paper Trading", "Backtest"])
        
        with tab1:
            live_df = strategies_df[strategies_df["account_type"] == "live"] if not strategies_df.empty else pd.DataFrame()
            if not live_df.empty:
                render_strategy_list(live_df, "Live Trading")
            else:
                st.info("No live trading strategies available.")
        
        with tab2:
            paper_df = strategies_df[strategies_df["account_type"] == "paper"] if not strategies_df.empty else pd.DataFrame()
            if not paper_df.empty:
                render_strategy_list(paper_df, "Paper Trading")
            else:
                st.info("No paper trading strategies available.")
        
        with tab3:
            backtest_df = strategies_df[strategies_df["account_type"] == "backtest"] if not strategies_df.empty else pd.DataFrame()
            if not backtest_df.empty:
                render_strategy_list(backtest_df, "Backtest")
            else:
                st.info("No backtest strategies available.")
    else:
        # If specific account type selected, only show that type
        filtered_df = strategies_df[strategies_df["account_type"] == data_filter] if not strategies_df.empty else pd.DataFrame()
        if not filtered_df.empty:
            render_strategy_list(filtered_df, account_type)
        else:
            st.info(f"No strategies available for {account_type}.")
