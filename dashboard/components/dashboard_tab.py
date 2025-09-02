"""
Dashboard Tab Component

This module renders the main Dashboard tab of the trading platform, following a professional,
institutional-grade design with dark theme, card-based components, and clear data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import random
from plotly.subplots import make_subplots

# Import UI styles
from dashboard.ui_styles import (
    ThemeMode, UIColors, UIEffects, UITypography, UISpacing,
    create_card, create_metric_card, format_currency, format_percentage,
    theme_plotly_chart
)

def create_portfolio_section(position_manager=None, broker_manager=None):
    """Creates the Live Portfolio section with metrics and charts"""
    
    st.markdown("<h2>Live Portfolio</h2>", unsafe_allow_html=True)
    
    # Try to get real portfolio data
    portfolio_value = 125750.43  # Default value
    daily_change = 1245.67  # Default value
    previous_value = portfolio_value - daily_change  # Default value
    realized_pnl = 8453.21  # Default value
    previous_pnl = 7982.45  # Default value
    unrealized_pnl = 3245.87  # Default value
    previous_unrealized = 2897.32  # Default value
    cash_balance = 42356.78  # Default value
    previous_cash = 45678.12  # Default value
    allocation_data = {
        'Asset Class': ['Stocks', 'Options', 'Forex', 'Crypto', 'Cash'],
        'Allocation': [35.2, 12.8, 8.5, 10.2, 33.3]
    }  # Default allocation
    
    # If we have actual position and broker managers, get real data
    if position_manager and broker_manager and hasattr(broker_manager, 'get_account_summary'):
        try:
            # Get portfolio data from session state if available
            if 'portfolio' in st.session_state:
                portfolio = st.session_state.portfolio
                portfolio_value = portfolio.get('total_value', portfolio_value)
                previous_value = portfolio.get('previous_value', previous_value)
                realized_pnl = portfolio.get('realized_pnl', realized_pnl)
                previous_pnl = portfolio.get('previous_realized_pnl', previous_pnl)
                unrealized_pnl = portfolio.get('unrealized_pnl', unrealized_pnl)
                previous_unrealized = portfolio.get('previous_unrealized_pnl', previous_unrealized)
                cash_balance = portfolio.get('cash_balance', cash_balance)
                previous_cash = portfolio.get('previous_cash', previous_cash)
            else:
                # Try to get data from broker manager
                account = broker_manager.get_account_summary()
                if account:
                    portfolio_value = account.get('portfolio_value', portfolio_value)
                    previous_value = account.get('previous_value', previous_value)
                    realized_pnl = account.get('realized_pnl', realized_pnl)
                    previous_pnl = account.get('previous_realized_pnl', previous_pnl)
                    unrealized_pnl = account.get('unrealized_pnl', unrealized_pnl)
                    previous_unrealized = account.get('previous_unrealized_pnl', previous_unrealized)
                    cash_balance = account.get('cash_balance', cash_balance)
                    previous_cash = account.get('previous_cash', previous_cash)
            
            # Get real allocation data if available
            if 'positions' in st.session_state and position_manager:
                positions = st.session_state.positions
                
                # Calculate allocations by asset class
                asset_classes = {'Stocks': 0, 'Options': 0, 'Forex': 0, 'Crypto': 0, 'Cash': cash_balance}
                
                for symbol, position in positions.items():
                    asset_class = position.get('asset_class', 'Stocks')  # Default to Stocks if not specified
                    market_value = position.get('market_value', 0)
                    
                    if asset_class in asset_classes:
                        asset_classes[asset_class] += market_value
                
                # Convert to percentages
                total = sum(asset_classes.values())
                allocation_percentages = []
                for asset_class, value in asset_classes.items():
                    allocation_percentages.append((value / total * 100) if total > 0 else 0)
                
                allocation_data = {
                    'Asset Class': list(asset_classes.keys()),
                    'Allocation': allocation_percentages
                }
        except Exception as e:
            st.error(f"Error retrieving portfolio data: {str(e)}")
            # Use default values if there's an error
            pass
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card("Total Portfolio Value", portfolio_value, previous_value),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card("Realized P&L (YTD)", realized_pnl, previous_pnl),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_metric_card("Unrealized P&L", unrealized_pnl, previous_unrealized),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_metric_card("Cash Balance", cash_balance, previous_cash),
            unsafe_allow_html=True
        )
    
    # Portfolio allocation chart
    colors = UIColors.Dark if st.session_state.theme_mode == ThemeMode.DARK else UIColors.Light
    
    fig = go.Figure(data=[go.Pie(
        labels=allocation_data['Asset Class'],
        values=allocation_data['Allocation'],
        hole=.4,
        textinfo='label+percent',
        marker=dict(
            colors=[colors.ACCENT_PRIMARY, colors.ACCENT_SECONDARY, 
                   colors.SUCCESS, colors.WARNING, colors.INFO],
            line=dict(color=colors.BG_CARD, width=1)
        )
    )])
    
    fig = theme_plotly_chart(fig, st.session_state.theme_mode)
    fig.update_layout(
        title="Portfolio Allocation",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_alerts_section(risk_manager=None):
    """Creates the Alerts & Messages section with status indicators"""
    
    st.markdown("<h2>Alerts & Messages</h2>", unsafe_allow_html=True)
    
    # Get alerts from session state if available, otherwise use sample data
    if 'alerts' in st.session_state:
        alerts = st.session_state.alerts
    else:
        # Generate sample alerts
        alerts = [
            {"type": "success", "title": "Trade Executed", "message": "BUY 100 AAPL @ $198.45", "time": "13:42:05"},
            {"type": "warning", "title": "Risk Threshold", "message": "Portfolio volatility approaching upper limit", "time": "13:30:22"},
            {"type": "error", "title": "API Connection Failed", "message": "Reconnecting to secondary data source", "time": "12:55:18"},
            {"type": "success", "title": "Strategy Activated", "message": "Gap Trading Strategy now active", "time": "09:30:00"},
            {"type": "warning", "title": "Position Warning", "message": "TSLA position exceeding 5% allocation", "time": "11:15:47"}
        ]
    
    # If risk manager is available, try to get active risk alerts
    if risk_manager and hasattr(risk_manager, 'get_active_alerts'):
        try:
            risk_alerts = risk_manager.get_active_alerts()
            if risk_alerts:
                # Convert risk alerts to our format and add to the beginning of the list
                for alert in risk_alerts:
                    alerts.insert(0, {
                        "type": alert.get('level', 'warning'),
                        "title": alert.get('title', 'Risk Alert'),
                        "message": alert.get('message', ''),
                        "time": alert.get('timestamp', datetime.datetime.now().strftime("%H:%M:%S"))
                    })
        except Exception as e:
            st.error(f"Error retrieving risk alerts: {str(e)}")
    
    # Filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        filter_options = st.multiselect(
            "Filter alerts by type:",
            ["Success", "Warning", "Error"],
            default=["Success", "Warning", "Error"]
        )
    
    with col2:
        st.write("")
        st.write("")
        show_all = st.checkbox("Show all alerts", value=False)
    
    with col3:
        st.write("")
        st.write("")
        if st.button("Clear all"):
            # Clear alerts in session state if button is clicked
            if 'alerts' in st.session_state:
                st.session_state.alerts = []
                st.rerun()
    
    # Check if we have any alerts to display
    if not alerts:
        st.info("No alerts to display")
    else:
        # Display alerts with proper styling
        for alert in alerts:
            status_class = alert["type"].lower()
            if status_class.capitalize() in filter_options or show_all:
                # Map alert type to color constant
                alert_type_upper = alert["type"].upper()
                if alert_type_upper == "SUCCESS":
                    color_attr = "SUCCESS"
                elif alert_type_upper == "WARNING":
                    color_attr = "WARNING"
                elif alert_type_upper == "ERROR":
                    color_attr = "ERROR"
                else:
                    color_attr = "INFO"
                
                alert_html = f"""
                <div class="card" style="border-left: 4px solid {getattr(UIColors.Dark, color_attr)}; padding: 12px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span class="status-dot {status_class}"></span>
                            <strong>{alert["title"]}</strong>
                        </div>
                        <small style="color: {UIColors.Dark.TEXT_TERTIARY};">{alert["time"]}</small>
                    </div>
                    <p style="margin: 8px 0 0 18px;">{alert["message"]}</p>
                </div>
                """
                st.markdown(alert_html, unsafe_allow_html=True)

def create_news_projections_section():
    """Creates the News & Projections of Current Live Trades section"""
    
    st.markdown("<h2>News & Projections</h2>", unsafe_allow_html=True)
    
    # Tab selection
    tab1, tab2 = st.tabs(["Market News", "Trade Projections"])
    
    with tab1:
        # Market News
        news_items = [
            {
                "title": "Fed Signals Potential Rate Cut in September",
                "source": "Financial Times",
                "timestamp": "2025-05-04 09:15",
                "impact": "high",
                "content": "The Federal Reserve has signaled a potential interest rate cut in September, citing improving inflation data and steady economic growth.",
                "assessment": "Bullish for equities, particularly growth stocks. Bearish for USD.",
                "action": "Monitor growth stock opportunities in tech and consumer discretionary."
            },
            {
                "title": "Tech Earnings Exceed Expectations",
                "source": "Bloomberg",
                "timestamp": "2025-05-04 10:30",
                "impact": "medium",
                "content": "Major tech companies reported Q1 earnings above analyst expectations, with cloud services showing particularly strong growth.",
                "assessment": "Positive for tech sector, may lift broader market sentiment.",
                "action": "Consider increasing tech allocation in quantitative strategies."
            },
            {
                "title": "Oil Prices Stabilize After Recent Volatility",
                "source": "Reuters",
                "timestamp": "2025-05-04 11:45",
                "impact": "medium",
                "content": "Oil prices have stabilized around $78/barrel after weeks of volatility, as supply concerns ease and demand remains steady.",
                "assessment": "Neutral for energy sector, positive for transportation.",
                "action": "No immediate action needed for energy positions."
            }
        ]
        
        # Display news cards with professional styling
        for news in news_items:
            impact_color = {
                "high": UIColors.Dark.ERROR,
                "medium": UIColors.Dark.WARNING,
                "low": UIColors.Dark.SUCCESS
            }.get(news["impact"], UIColors.Dark.INFO)
            
            news_card = f"""
            <div class="card" style="background-color: {UIColors.Dark.BG_TERTIARY}; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0;">{news["title"]}</h4>
                    <div>
                        <span style="background-color: {impact_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">
                            {news["impact"].upper()} IMPACT
                        </span>
                        <small style="color: {UIColors.Dark.TEXT_TERTIARY};">{news["timestamp"]}</small>
                    </div>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <small style="color: {UIColors.Dark.TEXT_SECONDARY};">Source: {news["source"]}</small>
                </div>
                <p>{news["content"]}</p>
                <div style="background-color: {UIColors.Dark.BG_SECONDARY}; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <div><strong>Impact Assessment:</strong> {news["assessment"]}</div>
                    <div><strong>Suggested Action:</strong> {news["action"]}</div>
                </div>
            </div>
            """
            st.markdown(news_card, unsafe_allow_html=True)
    
    with tab2:
        # Trade Projections
        st.markdown("<h3>Current Live Trades Projection</h3>", unsafe_allow_html=True)
        
        # Generate sample active trades
        active_trades = [
            {"symbol": "AAPL", "type": "Long", "entry": 198.45, "current": 202.34, "target": 215.00, "stop": 190.00, "pnl": 1.96},
            {"symbol": "EURUSD", "type": "Short", "entry": 1.0825, "current": 1.0798, "target": 1.0750, "stop": 1.0875, "pnl": 0.25},
            {"symbol": "TSLA", "type": "Long", "entry": 178.23, "current": 175.45, "target": 195.00, "stop": 170.00, "pnl": -1.56},
            {"symbol": "XOM", "type": "Long", "entry": 105.78, "current": 108.92, "target": 115.00, "stop": 100.00, "pnl": 2.97}
        ]
        
        # Create projection chart
        fig = make_subplots(rows=len(active_trades), cols=1, 
                          subplot_titles=[f"{t['symbol']} ({t['type']})" for t in active_trades],
                          vertical_spacing=0.05)
        
        for i, trade in enumerate(active_trades):
            # Calculate bar ranges
            entry_to_current = abs(trade["current"] - trade["entry"])
            current_to_target = abs(trade["target"] - trade["current"])
            current_to_stop = abs(trade["stop"] - trade["current"])
            
            # Determine colors based on trade type and position
            if trade["type"] == "Long":
                entry_color = UIColors.Dark.NEUTRAL
                target_color = UIColors.Dark.PROFIT
                stop_color = UIColors.Dark.LOSS
            else:
                entry_color = UIColors.Dark.NEUTRAL
                target_color = UIColors.Dark.PROFIT
                stop_color = UIColors.Dark.LOSS
            
            # Add bars
            fig.add_trace(
                go.Bar(
                    x=["Entry to Current", "Current to Target", "Current to Stop"],
                    y=[entry_to_current, current_to_target, current_to_stop],
                    text=[f"{trade['entry']} → {trade['current']}", 
                         f"{trade['current']} → {trade['target']}", 
                         f"{trade['current']} → {trade['stop']}"],
                    textposition="auto",
                    marker_color=[entry_color, target_color, stop_color],
                    hoverinfo="text",
                ),
                row=i+1, col=1
            )
            
            # Add annotations for PnL
            pnl_color = UIColors.Dark.PROFIT if trade["pnl"] >= 0 else UIColors.Dark.LOSS
            pnl_text = f"+{trade['pnl']}%" if trade["pnl"] >= 0 else f"{trade['pnl']}%"
            
            fig.add_annotation(
                x=2.5, y=max(entry_to_current, current_to_target, current_to_stop) * 0.9,
                text=f"P&L: {pnl_text}",
                font=dict(color=pnl_color, size=14),
                showarrow=False,
                row=i+1, col=1
            )
        
        # Theme and layout
        fig = theme_plotly_chart(fig, st.session_state.theme_mode)
        fig.update_layout(
            showlegend=False,
            height=150 * len(active_trades),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_ai_assistant_section(ai_coordinator=None):
    """Creates the AI Chat Assistant section with Minerva integration"""
    
    st.markdown("<h2>AI Chat Assistant</h2>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I'm Minerva, your AI trading assistant. How can I help you today?"}
        ]
    
    # Chat container with custom styling
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                message_html = f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                    <div style="background-color: {UIColors.Dark.ACCENT_PRIMARY}; color: white; padding: 10px 14px; border-radius: 18px 18px 0 18px; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """
            else:
                message_html = f"""
                <div style="display: flex; margin-bottom: 12px;">
                    <div style="background-color: {UIColors.Dark.BG_TERTIARY}; padding: 10px 14px; border-radius: 18px 18px 18px 0; max-width: 80%;">
                        {message["content"]}
                    </div>
                </div>
                """
            st.markdown(message_html, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Message Minerva", key="user_message", label_visibility="collapsed")
    
    with col2:
        st.write("")
        send_button = st.button("Send", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from Minerva AI or use sample responses if not available
        if ai_coordinator and hasattr(ai_coordinator, 'process_message'):
            try:
                # Try to get contextual information
                context = {}
                
                # Add portfolio information if available
                if 'portfolio' in st.session_state:
                    context['portfolio'] = st.session_state.portfolio
                
                # Add positions if available
                if 'positions' in st.session_state:
                    context['positions'] = st.session_state.positions
                
                # Add recent alerts if available
                if 'alerts' in st.session_state:
                    context['alerts'] = st.session_state.alerts[:5]  # Only provide the 5 most recent alerts
                
                # Process the message through Minerva
                response = ai_coordinator.process_message(
                    message=user_input,
                    chat_history=st.session_state.chat_history,
                    context=context
                )
                
                ai_response = response.get('content', 
                                         "I'm having trouble processing your request right now. Please try again later.")
            except Exception as e:
                st.error(f"Error connecting to Minerva AI: {str(e)}")
                # Fallback to demo responses
                ai_response = "I'm having trouble accessing my full capabilities right now. Let me provide a simpler response.\n\n" + get_demo_response(user_input)
        else:
            # Use demo responses if Minerva is not available
            ai_response = get_demo_response(user_input)
        
        # Add AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Clear the input box and rerun to show the updated chat
        st.rerun()

def get_demo_response(query):
    """Get a demo response based on the query content"""
    # Simple keyword matching for demo responses
    query = query.lower()
    
    if any(word in query for word in ['portfolio', 'holdings', 'positions']):
        return "Your portfolio currently has a good diversification across sectors, with a slight overweight in technology. Your largest position is AAPL at 8.2% of your portfolio."
    
    elif any(word in query for word in ['market', 'trend', 'outlook']):
        return "The market is showing mixed signals currently. Economic indicators suggest potential volatility ahead, but earnings have been generally positive this quarter."
    
    elif any(word in query for word in ['strategy', 'recommend', 'suggestion']):
        return "Based on your risk profile and current market conditions, I would suggest considering a more defensive positioning. You might want to look at reducing exposure to high-beta tech stocks and adding some quality dividend payers."
    
    elif any(word in query for word in ['risk', 'exposure', 'correlation']):
        return "Your portfolio's current risk metrics show a beta of 1.2 relative to the S&P 500, indicating slightly higher volatility than the market. Your sector correlations are relatively high, suggesting you could benefit from more diversification."
    
    elif any(word in query for word in ['performance', 'return', 'profit']):
        return "Your portfolio has returned 8.2% year-to-date, outperforming the S&P 500 by 1.3%. Your best performing position is NVDA with a 45% gain, while your worst is XOM with a 5% loss."
    
    else:
        # Default responses for other queries
        responses = [
            "I've analyzed your portfolio and noticed your tech exposure is high relative to your risk profile. Consider diversifying.",
            "The recent news about interest rates might impact your forex positions. Would you like me to analyze potential effects?",
            "Based on your trading history, you tend to exit profitable trades too early. Consider setting trailing stops instead.",
            "I've detected a potential gap trading opportunity for tomorrow's open. Would you like details?",
            "Your current win rate is 62% with an average risk-reward ratio of 1.5. This is above your 3-month average."
        ]
        return random.choice(responses)

def render_dashboard_tab(position_manager=None, broker_manager=None, risk_manager=None, ai_coordinator=None):
    """Renders the complete Dashboard tab"""
    
    st.title("Trading Dashboard")
    
    # Create top row with two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Live Portfolio section (left top)
        create_portfolio_section(position_manager, broker_manager)
    
    with col2:
        # Alerts & Messages section (right top)
        create_alerts_section(risk_manager)
    
    # Create bottom row 
    # News & Projections section (bottom)
    create_news_projections_section()
    
    # AI Chat Assistant (bottom)
    create_ai_assistant_section(ai_coordinator)
