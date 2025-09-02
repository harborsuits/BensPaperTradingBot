"""
Developer Tab Component

This module renders the Developer tab of the trading platform, providing real-time
monitoring of the event system, configuration of trading modes, and system diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import random
import json
from plotly.subplots import make_subplots

# Import UI styles
from dashboard.ui_styles import (
    ThemeMode, UIColors, UIEffects, UITypography, UISpacing,
    create_card, create_metric_card, format_currency, format_percentage,
    theme_plotly_chart
)

def create_event_system_section():
    """Creates the Event System monitoring section"""
    
    st.markdown("<h2>Event System</h2>", unsafe_allow_html=True)
    
    # Tabs for different event system components
    event_tabs = st.tabs(["Event Bus", "Message Queues", "Channels"])
    
    with event_tabs[0]:
        # Event Bus monitoring
        st.markdown("<h3>Active Event Listeners</h3>", unsafe_allow_html=True)
        
        # Sample event listeners
        event_listeners = [
            {"id": "EL001", "event_type": "MARKET_DATA", "component": "StrategyManager", "status": "Active", "count": 542},
            {"id": "EL002", "event_type": "ORDER_UPDATE", "component": "PositionManager", "status": "Active", "count": 78},
            {"id": "EL003", "event_type": "SIGNAL_GENERATED", "component": "StrategyOrchestrator", "status": "Active", "count": 124},
            {"id": "EL004", "event_type": "PORTFOLIO_UPDATE", "component": "RiskManager", "status": "Active", "count": 45},
            {"id": "EL005", "event_type": "RISK_ALERT", "component": "NotificationSystem", "status": "Active", "count": 12},
            {"id": "EL006", "event_type": "BROKER_CONNECTION", "component": "BrokerManager", "status": "Inactive", "count": 0},
        ]
        
        # Create a DataFrame
        df_listeners = pd.DataFrame(event_listeners)
        
        # Add styling
        def highlight_status(val):
            if val == 'Active':
                return f'background-color: {UIColors.Dark.SUCCESS}; color: white;'
            else:
                return f'background-color: {UIColors.Dark.ERROR}; color: white;'
        
        # Apply the styling and display
        styled_df = df_listeners.style.applymap(highlight_status, subset=['status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Recent events
        st.markdown("<h3>Recent Events</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Sample recent events
            recent_events = [
                {"timestamp": "13:45:05", "type": "MARKET_DATA", "source": "AlpacaDataProvider", "content": "Price update for AAPL: $202.34"},
                {"timestamp": "13:44:58", "type": "SIGNAL_GENERATED", "source": "GapTradingStrategy", "content": "Buy signal generated for NVDA"},
                {"timestamp": "13:44:45", "type": "ORDER_UPDATE", "source": "TradierBroker", "content": "Order filled: Buy 100 AAPL @ $202.34"},
                {"timestamp": "13:44:22", "type": "RISK_ALERT", "source": "RiskManager", "content": "Portfolio volatility increased to 18.5%"},
                {"timestamp": "13:44:10", "type": "PORTFOLIO_UPDATE", "source": "PositionManager", "content": "New position: Long 100 AAPL @ $202.34"}
            ]
            
            for event in recent_events:
                event_type_color = {
                    "MARKET_DATA": UIColors.Dark.INFO,
                    "SIGNAL_GENERATED": UIColors.Dark.SUCCESS,
                    "ORDER_UPDATE": UIColors.Dark.ACCENT_PRIMARY,
                    "RISK_ALERT": UIColors.Dark.WARNING,
                    "PORTFOLIO_UPDATE": UIColors.Dark.ACCENT_SECONDARY,
                    "BROKER_CONNECTION": UIColors.Dark.ERROR
                }.get(event["type"], UIColors.Dark.NEUTRAL)
                
                event_html = f"""
                <div style="display: flex; margin-bottom: 8px; border-left: 3px solid {event_type_color}; padding: 8px; background-color: {UIColors.Dark.BG_TERTIARY};">
                    <div style="width: 80px; color: {UIColors.Dark.TEXT_TERTIARY};">{event["timestamp"]}</div>
                    <div style="width: 160px; color: {event_type_color};">{event["type"]}</div>
                    <div style="width: 180px; color: {UIColors.Dark.TEXT_SECONDARY};">{event["source"]}</div>
                    <div style="flex: 1;">{event["content"]}</div>
                </div>
                """
                st.markdown(event_html, unsafe_allow_html=True)
        
        with col2:
            # Event type distribution
            event_types = ["MARKET_DATA", "SIGNAL_GENERATED", "ORDER_UPDATE", "RISK_ALERT", "PORTFOLIO_UPDATE", "BROKER_CONNECTION"]
            event_counts = [542, 124, 78, 12, 45, 8]
            
            colors = UIColors.Dark if st.session_state.theme_mode == ThemeMode.DARK else UIColors.Light
            
            fig = go.Figure(data=[go.Pie(
                labels=event_types,
                values=event_counts,
                hole=.3,
                textinfo='percent',
                textfont_size=10,
                marker=dict(
                    colors=[colors.INFO, colors.SUCCESS, colors.ACCENT_PRIMARY, 
                           colors.WARNING, colors.ACCENT_SECONDARY, colors.ERROR],
                    line=dict(color=colors.BG_CARD, width=1)
                )
            )])
            
            fig = theme_plotly_chart(fig, st.session_state.theme_mode)
            fig.update_layout(
                title="Event Distribution",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with event_tabs[1]:
        # Message Queue monitoring
        st.markdown("<h3>Active Message Queues</h3>", unsafe_allow_html=True)
        
        # Sample message queues
        message_queues = [
            {"name": "MarketData", "pending_messages": 8, "processed_messages": 1254, "avg_latency_ms": 12.5, "status": "Healthy"},
            {"name": "OrderExecution", "pending_messages": 2, "processed_messages": 145, "avg_latency_ms": 8.2, "status": "Healthy"},
            {"name": "RiskUpdates", "pending_messages": 0, "processed_messages": 78, "avg_latency_ms": 15.1, "status": "Healthy"},
            {"name": "Notifications", "pending_messages": 3, "processed_messages": 42, "avg_latency_ms": 5.3, "status": "Healthy"}
        ]
        
        for queue in message_queues:
            status_color = UIColors.Dark.SUCCESS if queue["status"] == "Healthy" else UIColors.Dark.WARNING
            
            queue_html = f"""
            <div class="card" style="margin-bottom: 12px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h4 style="margin: 0;">{queue["name"]} Queue</h4>
                    <div style="display: flex; align-items: center;">
                        <span class="status-dot success"></span>
                        <span style="color: {status_color};">{queue["status"]}</span>
                    </div>
                </div>
                
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 24px; font-weight: bold;">
                            {queue["pending_messages"]}
                        </div>
                        <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                            Pending
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold;">
                            {queue["processed_messages"]}
                        </div>
                        <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                            Processed
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold;">
                            {queue["avg_latency_ms"]} ms
                        </div>
                        <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                            Avg Latency
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 16px;">
                    <div style="font-size: 14px; margin-bottom: 4px;">Queue Processing Rate</div>
                    <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                        <div style="width: 85%; height: 100%; background-color: {UIColors.Dark.ACCENT_PRIMARY}; border-radius: 4px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY}; margin-top: 4px;">
                        <span>0</span>
                        <span>85 msg/sec</span>
                        <span>100</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(queue_html, unsafe_allow_html=True)
    
    with event_tabs[2]:
        # Channels monitoring
        st.markdown("<h3>Active Communication Channels</h3>", unsafe_allow_html=True)
        
        # Sample channels
        channels = [
            {"name": "StrategyData", "subscribers": 4, "messages_per_min": 120, "status": "Active"},
            {"name": "MarketUpdates", "subscribers": 6, "messages_per_min": 75, "status": "Active"},
            {"name": "OrderFlow", "subscribers": 3, "messages_per_min": 25, "status": "Active"},
            {"name": "RiskMonitoring", "subscribers": 2, "messages_per_min": 15, "status": "Active"}
        ]
        
        col1, col2 = st.columns(2)
        
        for i, channel in enumerate(channels):
            with col1 if i % 2 == 0 else col2:
                channel_html = f"""
                <div class="card" style="margin-bottom: 12px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="margin: 0;">{channel["name"]}</h4>
                        <div style="display: flex; align-items: center;">
                            <span class="status-dot success"></span>
                            <span style="color: {UIColors.Dark.SUCCESS};">{channel["status"]}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px;">
                        <div>
                            <div style="font-size: 24px; font-weight: bold;">
                                {channel["subscribers"]}
                            </div>
                            <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                                Subscribers
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 24px; font-weight: bold;">
                                {channel["messages_per_min"]}
                            </div>
                            <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                                Msgs/Min
                            </div>
                        </div>
                    </div>
                    
                    <div style="font-size: 14px; margin-bottom: 4px;">Subscribers:</div>
                    <div style="background-color: {UIColors.Dark.BG_SECONDARY}; padding: 8px; border-radius: 4px; font-family: {UITypography.FONT_MONO}; font-size: 12px;">
                        {generate_subscriber_list(channel["name"], channel["subscribers"])}
                    </div>
                </div>
                """
                st.markdown(channel_html, unsafe_allow_html=True)
        
        # Function to generate mock subscriber list
        def generate_subscriber_list(channel_name, count):
            components = ["StrategyManager", "PositionManager", "RiskManager", "OrderManager", 
                         "NotificationSystem", "DataAnalyzer", "MarketDataProvider", "BrokerManager"]
            
            subscribers = []
            for i in range(count):
                if i < len(components):
                    subscribers.append(components[i])
                else:
                    subscribers.append(f"Component{i+1}")
            
            return ", ".join(subscribers)

def create_trading_modes_section():
    """Creates the Trading Modes configuration section"""
    
    st.markdown("<h2>Trading Modes</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Current mode selection
        st.markdown("<h3>Current Mode</h3>", unsafe_allow_html=True)
        
        trading_modes = ["Standard Trading", "Paper Trading", "Backtest", "Simulation", "Development"]
        selected_mode = st.selectbox("Active Trading Mode", trading_modes, index=0)
        
        # Mode description
        mode_descriptions = {
            "Standard Trading": "Live trading with real money. All safeguards and risk controls active.",
            "Paper Trading": "Simulated trading with real market data but virtual capital.",
            "Backtest": "Historical testing against past market data.",
            "Simulation": "Monte Carlo simulation with randomized market conditions.",
            "Development": "For strategy development with mock data and no execution."
        }
        
        mode_html = f"""
        <div class="card" style="margin-top: 16px; padding: 16px; background-color: {UIColors.Dark.BG_TERTIARY};">
            <p style="margin: 0 0 12px 0;">{mode_descriptions[selected_mode]}</p>
            
            <div style="font-size: 14px; margin-bottom: 8px;"><strong>Mode Status:</strong></div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span class="status-dot success"></span>
                <span>Active</span>
            </div>
            <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY}; margin-bottom: 12px;">
                Activated: 2025-05-04 09:30:15
            </div>
            
            <button style="width: 100%; background-color: {UIColors.Dark.WARNING}; color: white; border: none; padding: 8px; border-radius: 4px; margin-top: 8px;">
                Switch Trading Mode
            </button>
        </div>
        """
        st.markdown(mode_html, unsafe_allow_html=True)
    
    with col2:
        # Mode configuration
        st.markdown("<h3>Mode Configuration</h3>", unsafe_allow_html=True)
        
        # Tab selection for different modes
        config_tabs = st.tabs(["Order Execution", "Risk Controls", "Data Sources", "Capital Allocation"])
        
        with config_tabs[0]:
            # Order Execution settings
            st.markdown("<h4>Order Execution Settings</h4>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.selectbox("Execution Algorithm", ["Market", "Limit", "TWAP", "VWAP", "Iceberg"])
                st.slider("Urgency Level", min_value=1, max_value=10, value=5)
                st.checkbox("Enable Smart Order Routing", value=True)
            
            with col_b:
                st.number_input("Max Slippage (%)", min_value=0.01, max_value=2.0, value=0.10, step=0.01)
                st.number_input("Order Timeout (sec)", min_value=5, max_value=300, value=30, step=5)
                st.checkbox("Auto-retry Failed Orders", value=True)
        
        with config_tabs[1]:
            # Risk Control settings
            st.markdown("<h4>Risk Control Settings</h4>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.number_input("Max Position Size (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
                st.number_input("Max Drawdown (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
                st.checkbox("Auto-hedge Excessive Risk", value=False)
            
            with col_b:
                st.number_input("Position Correlation Limit", min_value=0.3, max_value=0.9, value=0.7, step=0.05)
                st.number_input("VaR Limit (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
                st.checkbox("Enable Circuit Breaker", value=True)
        
        with config_tabs[2]:
            # Data Sources settings
            st.markdown("<h4>Data Source Settings</h4>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.selectbox("Primary Market Data", ["Alpaca", "Tradier", "Interactive Brokers", "Polygon"])
                st.selectbox("Primary News Source", ["Alpha Vantage", "NewsData.io", "Marketaux", "Bloomberg"])
                st.checkbox("Enable Data Fallbacks", value=True)
            
            with col_b:
                st.selectbox("Backup Market Data", ["Polygon", "Alpaca", "Finnhub", "None"])
                st.selectbox("Alternative Data", ["Sentiment Analysis", "Social Media", "Options Flow", "None"])
                st.number_input("Data Refresh Rate (sec)", min_value=1, max_value=60, value=5, step=1)
        
        with config_tabs[3]:
            # Capital Allocation settings
            st.markdown("<h4>Capital Allocation Settings</h4>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.selectbox("Allocation Strategy", ["Equal Weighting", "Risk Parity", "Kelly Criterion", "Fixed Percentage"])
                st.number_input("Reserve Cash (%)", min_value=0.0, max_value=50.0, value=10.0, step=5.0)
                st.checkbox("Dynamic Allocation", value=True)
            
            with col_b:
                st.number_input("Max Single Strategy (%)", min_value=5.0, max_value=50.0, value=20.0, step=5.0)
                st.number_input("Rebalance Threshold (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
                st.checkbox("Auto-rebalance", value=True)

def create_system_diagnostics_section():
    """Creates the System Diagnostics section"""
    
    st.markdown("<h2>System Diagnostics</h2>", unsafe_allow_html=True)
    
    # Create columns for different diagnostic displays
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # System metrics
        st.markdown("<h3>System Metrics</h3>", unsafe_allow_html=True)
        
        metrics_html = f"""
        <div class="card" style="padding: 16px;">
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>CPU Usage</strong>
                    <span>32%</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 32%; height: 100%; background-color: {UIColors.Dark.SUCCESS}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>Memory Usage</strong>
                    <span>1.4 GB (45%)</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 45%; height: 100%; background-color: {UIColors.Dark.SUCCESS}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>Network I/O</strong>
                    <span>3.2 MB/s</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 28%; height: 100%; background-color: {UIColors.Dark.INFO}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>Database I/O</strong>
                    <span>0.8 MB/s</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 18%; height: 100%; background-color: {UIColors.Dark.INFO}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>Uptime</strong>
                    <span>3d 12h 45m</span>
                </div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    with col2:
        # Event metrics
        st.markdown("<h3>Event Metrics</h3>", unsafe_allow_html=True)
        
        # Generate event metrics data
        event_rates = {
            "Market Data": 85,
            "Order Updates": 12,
            "Portfolio Updates": 8,
            "Risk Alerts": 2,
            "Strategy Signals": 15
        }
        
        # Create bar chart
        fig = go.Figure()
        
        colors = UIColors.Dark if st.session_state.theme_mode == ThemeMode.DARK else UIColors.Light
        
        fig.add_trace(go.Bar(
            x=list(event_rates.values()),
            y=list(event_rates.keys()),
            orientation='h',
            marker=dict(
                color=[colors.INFO, colors.ACCENT_PRIMARY, colors.ACCENT_SECONDARY, colors.WARNING, colors.SUCCESS],
            )
        ))
        
        fig = theme_plotly_chart(fig, st.session_state.theme_mode)
        fig.update_layout(
            title="Events Per Second",
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Events/sec",
            yaxis_title=None
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Event latency
        latency_html = f"""
        <div class="card" style="padding: 16px;">
            <h4 style="margin-top: 0;">Event Processing Latency</h4>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 20px; font-weight: bold;">5.2 ms</div>
                    <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">Average</div>
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: bold;">1.8 ms</div>
                    <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">Minimum</div>
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: bold;">28.5 ms</div>
                    <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">Maximum</div>
                </div>
            </div>
        </div>
        """
        st.markdown(latency_html, unsafe_allow_html=True)
    
    with col3:
        # Architecture visualization
        st.markdown("<h3>System Architecture</h3>", unsafe_allow_html=True)
        
        # System architecture visualization
        system_arch = {
            "nodes": [
                {"id": "data", "label": "Data Sources", "group": 1},
                {"id": "event", "label": "Event Bus", "group": 2},
                {"id": "strategy", "label": "Strategies", "group": 3},
                {"id": "risk", "label": "Risk Management", "group": 4},
                {"id": "order", "label": "Order Management", "group": 5},
                {"id": "broker", "label": "Broker Integration", "group": 6},
                {"id": "portfolio", "label": "Portfolio Manager", "group": 4},
                {"id": "ui", "label": "UI", "group": 7}
            ],
            "links": [
                {"source": "data", "target": "event", "value": 10},
                {"source": "event", "target": "strategy", "value": 8},
                {"source": "strategy", "target": "event", "value": 5},
                {"source": "event", "target": "risk", "value": 5},
                {"source": "risk", "target": "event", "value": 3},
                {"source": "event", "target": "order", "value": 5},
                {"source": "order", "target": "event", "value": 3},
                {"source": "order", "target": "broker", "value": 5},
                {"source": "broker", "target": "order", "value": 3},
                {"source": "event", "target": "portfolio", "value": 3},
                {"source": "portfolio", "target": "event", "value": 2},
                {"source": "event", "target": "ui", "value": 2}
            ]
        }
        
        # Convert to JSON for display
        system_arch_json = json.dumps(system_arch, indent=2)
        
        # Create styled display
        arch_html = f"""
        <div class="card" style="height: 335px; overflow: auto; font-family: {UITypography.FONT_MONO}; font-size: 12px; padding: 16px; background-color: {UIColors.Dark.BG_TERTIARY};">
            <pre style="margin: 0; color: {UIColors.Dark.TEXT_PRIMARY};">{system_arch_json}</pre>
        </div>
        """
        st.markdown(arch_html, unsafe_allow_html=True)

def render_developer_tab():
    """Renders the complete Developer tab"""
    
    st.title("Developer Dashboard")
    
    # Event System section
    create_event_system_section()
    
    # Trading Modes section
    create_trading_modes_section()
    
    # System Diagnostics section
    create_system_diagnostics_section()
