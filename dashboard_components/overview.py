"""
Overview Component for BensBot Dashboard
Displays combined P&L curves, risk posture, and comparative performance metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get equity curve data from MongoDB
def get_equity_data(db, account_types=None):
    """
    Retrieve equity curve data from MongoDB for specified account types
    Returns a dict with account types as keys and DataFrame of dates/equity as values
    """
    if db is None:
        return generate_mock_equity_data(account_types)
    
    result = {}
    try:
        # Default to all account types if none specified
        if account_types is None:
            account_types = ["paper", "live", "backtest"]
        
        for account_type in account_types:
            # Try to get from MongoDB
            equity_docs = list(db.equity_history.find(
                {"account_type": account_type}
            ).sort("date", 1).limit(90))  # Last 90 days
            
            if equity_docs:
                df = pd.DataFrame(equity_docs)
                df['date'] = pd.to_datetime(df['date'])
                result[account_type] = df[['date', 'equity']]
            else:
                # If no data found, generate mock data
                result[account_type] = generate_mock_equity_data([account_type])[account_type]
        
        return result
    except Exception as e:
        st.error(f"Error retrieving equity data: {e}")
        return generate_mock_equity_data(account_types)

# Generate mock equity data for development/testing
def generate_mock_equity_data(account_types=None):
    """Generate synthetic equity curves for development and testing"""
    if account_types is None:
        account_types = ["paper", "live", "backtest"]
    
    result = {}
    now = datetime.datetime.now()
    days = 90
    
    # Create date range
    dates = [now - datetime.timedelta(days=i) for i in range(days, 0, -1)]
    
    # Generate curves with different characteristics for each account type
    for account_type in account_types:
        # Set starting point and volatility based on account type
        if account_type == "live":
            starting_balance = 100000
            volatility = 0.007  # 0.7% daily volatility
            trend = 0.001  # 0.1% positive drift
        elif account_type == "paper":
            starting_balance = 100000
            volatility = 0.01  # 1% daily volatility
            trend = 0.002  # 0.2% positive drift
        else:  # backtest
            starting_balance = 100000
            volatility = 0.005  # 0.5% daily volatility
            trend = 0.003  # 0.3% positive drift
        
        # Generate equity curve
        equity = [starting_balance]
        for i in range(1, days):
            # Random walk with trend
            change = np.random.normal(trend, volatility) * equity[-1]
            new_value = max(0, equity[-1] + change)
            equity.append(new_value)
        
        # Create DataFrame
        result[account_type] = pd.DataFrame({
            'date': dates,
            'equity': equity,
            'account_type': account_type
        })
    
    return result

# Get risk posture data
def get_risk_posture(db):
    """Get current market regime and risk posture data"""
    if db is None:
        return generate_mock_risk_data()
    
    try:
        # Try to get from MongoDB
        risk_doc = db.market_context.find_one({"type": "risk_posture"})
        
        if risk_doc:
            return risk_doc
        else:
            return generate_mock_risk_data()
    except Exception as e:
        st.error(f"Error retrieving risk posture: {e}")
        return generate_mock_risk_data()

# Generate mock risk posture data
def generate_mock_risk_data():
    """Generate synthetic risk posture data for development and testing"""
    return {
        "market_regime": "NEUTRAL",
        "volatility_level": 0.65,  # 0-1 scale
        "trend_strength": 0.42,    # 0-1 scale
        "risk_appetite": 0.58,     # 0-1 scale
        "macro_conditions": "STABLE",
        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Get performance comparison data
def get_performance_comparison(db):
    """Get comparative performance metrics across account types"""
    if db is None:
        return generate_mock_performance_data()
    
    try:
        # Try to get from MongoDB
        perf_docs = list(db.performance_metrics.find({}))
        
        if perf_docs:
            # Convert to dictionary with account_type as key
            result = {}
            for doc in perf_docs:
                account_type = doc.get("account_type", "unknown")
                result[account_type] = doc
            return result
        else:
            return generate_mock_performance_data()
    except Exception as e:
        st.error(f"Error retrieving performance data: {e}")
        return generate_mock_performance_data()

# Generate mock performance data
def generate_mock_performance_data():
    """Generate synthetic performance comparison data for development and testing"""
    return {
        "live": {
            "account_type": "live",
            "win_rate": 62.5,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.42,
            "max_drawdown": -4.2,
            "avg_win": 580.25,
            "avg_loss": -320.50,
            "total_trades": 48,
            "pnl_pct": 8.3,
            "annual_return": 12.7
        },
        "paper": {
            "account_type": "paper",
            "win_rate": 58.7,
            "profit_factor": 1.65,
            "sharpe_ratio": 1.28,
            "max_drawdown": -5.8,
            "avg_win": 610.35,
            "avg_loss": -370.25,
            "total_trades": 126,
            "pnl_pct": 15.2,
            "annual_return": 18.5
        },
        "backtest": {
            "account_type": "backtest",
            "win_rate": 65.2,
            "profit_factor": 2.1,
            "sharpe_ratio": 1.85,
            "max_drawdown": -3.5,
            "avg_win": 550.80,
            "avg_loss": -280.65,
            "total_trades": 352,
            "pnl_pct": 22.7,
            "annual_return": 24.3
        }
    }

# Render the equity curve with time period selector
def render_equity_curve(equity_data):
    """Render interactive equity curve with account type toggling"""
    st.subheader("Equity Curve")
    
    # Time period selector
    time_periods = {
        "1W": 7,
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "YTD": (datetime.datetime.now() - datetime.datetime(datetime.datetime.now().year, 1, 1)).days,
        "1Y": 365,
        "ALL": 999
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_period = st.select_slider(
            "Time Period", 
            options=list(time_periods.keys()),
            value="3M"
        )
    
    with col2:
        # Account type toggles
        account_toggles = {}
        
        if "live" in equity_data:
            account_toggles["Live Trading"] = st.checkbox("Live Trading", value=True)
        
        if "paper" in equity_data:
            account_toggles["Paper Trading"] = st.checkbox("Paper Trading", value=True)
        
        if "backtest" in equity_data:
            account_toggles["Backtest"] = st.checkbox("Backtest", value=True)
    
    # Filter data by time period
    days_to_show = time_periods[selected_period]
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_show)
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Map display names to data keys
    name_to_key = {
        "Live Trading": "live",
        "Paper Trading": "paper",
        "Backtest": "backtest"
    }
    
    # Line colors by account type
    colors = {
        "live": "#1f77b4",  # Blue
        "paper": "#ff7f0e",  # Orange
        "backtest": "#2ca02c"  # Green
    }
    
    # Add lines for selected account types
    for display_name, is_visible in account_toggles.items():
        if is_visible:
            key = name_to_key[display_name]
            if key in equity_data:
                df = equity_data[key]
                df_filtered = df[df['date'] >= cutoff_date]
                
                if not df_filtered.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df_filtered["date"],
                            y=df_filtered["equity"],
                            mode="lines",
                            name=display_name,
                            line=dict(width=3, color=colors.get(key, "#000000"))
                        )
                    )
    
    # Enhance the chart layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="",
        yaxis_title="Equity ($)",
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

# Render risk posture gauge chart
def render_risk_posture(risk_data):
    """Render risk posture visualization with gauges"""
    st.subheader("Market Regime & Risk Posture")
    
    # Create a 2-column layout with 3 metrics in first column and gauge chart in second
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Market regime indicator
        regime = risk_data.get("market_regime", "NEUTRAL")
        regime_color = {
            "BULLISH": "green",
            "NEUTRAL": "blue",
            "BEARISH": "red",
            "VOLATILE": "orange"
        }.get(regime, "gray")
        
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; color: #666;">Current Market Regime</p>
            <p style="font-size: 1.5rem; font-weight: 700; color: {regime_color};">{regime}</p>
            <p style="font-size: 0.8rem; color: #888;">Last updated: {risk_data.get('last_update', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Macro conditions indicator
        macro = risk_data.get("macro_conditions", "STABLE")
        macro_color = {
            "EXPANSIONARY": "green",
            "STABLE": "blue",
            "CONTRACTIONARY": "red",
            "UNCERTAIN": "orange"
        }.get(macro, "gray")
        
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; color: #666;">Macro Conditions</p>
            <p style="font-size: 1.5rem; font-weight: 700; color: {macro_color};">{macro}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Strategy recommendations
        risk_appetite = risk_data.get("risk_appetite", 0.5)
        if risk_appetite > 0.66:
            recommendation = "Aggressive"
            rec_color = "green"
        elif risk_appetite > 0.33:
            recommendation = "Balanced"
            rec_color = "blue"
        else:
            recommendation = "Conservative"
            rec_color = "orange"
        
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem; color: #666;">Strategy Recommendation</p>
            <p style="font-size: 1.5rem; font-weight: 700; color: {rec_color};">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create gauge charts for risk metrics
        fig = make_subplots(
            rows=3, cols=1,
            specs=[[{"type": "indicator"}], [{"type": "indicator"}], [{"type": "indicator"}]],
            row_heights=[0.33, 0.33, 0.33],
            vertical_spacing=0.1
        )
        
        # Volatility gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data.get("volatility_level", 0.5) * 100,
                title={"text": "Volatility Level", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "royalblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 33], "color": "rgba(0, 250, 0, 0.3)"},
                        {"range": [33, 66], "color": "rgba(250, 250, 0, 0.3)"},
                        {"range": [66, 100], "color": "rgba(250, 0, 0, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": risk_data.get("volatility_level", 0.5) * 100
                    }
                },
                number={"suffix": "%"}
            ),
            row=1, col=1
        )
        
        # Trend strength gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data.get("trend_strength", 0.5) * 100,
                title={"text": "Trend Strength", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "royalblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 33], "color": "rgba(250, 0, 0, 0.3)"},
                        {"range": [33, 66], "color": "rgba(250, 250, 0, 0.3)"},
                        {"range": [66, 100], "color": "rgba(0, 250, 0, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": risk_data.get("trend_strength", 0.5) * 100
                    }
                },
                number={"suffix": "%"}
            ),
            row=2, col=1
        )
        
        # Risk appetite gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_data.get("risk_appetite", 0.5) * 100,
                title={"text": "Risk Appetite", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "royalblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 33], "color": "rgba(250, 0, 0, 0.3)"},
                        {"range": [33, 66], "color": "rgba(250, 250, 0, 0.3)"},
                        {"range": [66, 100], "color": "rgba(0, 250, 0, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": risk_data.get("risk_appetite", 0.5) * 100
                    }
                },
                number={"suffix": "%"}
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Render performance comparison
def render_performance_comparison(performance_data):
    """Render comparative metrics between account types"""
    st.subheader("Performance Comparison")
    
    # Create a 3-column layout for the three account types
    cols = st.columns(3)
    
    # Account type display names and order
    account_types = [
        {"key": "live", "name": "Live Trading", "color": "#1f77b4"},
        {"key": "paper", "name": "Paper Trading", "color": "#ff7f0e"},
        {"key": "backtest", "name": "Backtest", "color": "#2ca02c"}
    ]
    
    # Performance metrics to display
    metrics = [
        {"key": "win_rate", "name": "Win Rate", "format": "{:.1f}%", "compare": "higher"},
        {"key": "profit_factor", "name": "Profit Factor", "format": "{:.2f}", "compare": "higher"},
        {"key": "sharpe_ratio", "name": "Sharpe Ratio", "format": "{:.2f}", "compare": "higher"},
        {"key": "max_drawdown", "name": "Max Drawdown", "format": "{:.1f}%", "compare": "lower"},
        {"key": "total_trades", "name": "Total Trades", "format": "{}", "compare": "none"},
        {"key": "pnl_pct", "name": "Total P&L", "format": "{:.1f}%", "compare": "higher"},
        {"key": "annual_return", "name": "Annual Return", "format": "{:.1f}%", "compare": "higher"}
    ]
    
    # Render each account type column
    for i, account in enumerate(account_types):
        with cols[i]:
            account_key = account["key"]
            if account_key in performance_data:
                data = performance_data[account_key]
                
                # Header with background color
                st.markdown(f"""
                <div style="background-color: {account['color']}; padding: 10px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; font-size: 1.2rem;">{account['name']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                for metric in metrics:
                    value = data.get(metric['key'], 0)
                    formatted_value = metric['format'].format(value)
                    
                    st.markdown(f"""
                    <div style="border-bottom: 1px solid #eee; padding: 8px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-size: 0.9rem; color: #666;">{metric['name']}</span>
                            <span style="font-size: 0.9rem; font-weight: 600;">{formatted_value}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; font-size: 1.2rem; color: #666;">{account['name']}</h3>
                    <p style="margin-top: 5px; font-size: 0.8rem; color: #888;">No data available</p>
                </div>
                """, unsafe_allow_html=True)

# Main render function for this component
def render(db, account_type="All Accounts"):
    """Main render function for the Overview section"""
    # Get data based on account type selection
    account_filter = None
    if account_type != "All Accounts":
        if account_type == "Paper Trading":
            account_filter = ["paper"]
        elif account_type == "Live Trading":
            account_filter = ["live"]
        elif account_type == "Backtest":
            account_filter = ["backtest"]
    
    # Get data from MongoDB (or mock data if unavailable)
    equity_data = get_equity_data(db, account_filter)
    risk_data = get_risk_posture(db)
    performance_data = get_performance_comparison(db)
    
    # Render the components
    render_equity_curve(equity_data)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_risk_posture(risk_data)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_performance_comparison(performance_data)
