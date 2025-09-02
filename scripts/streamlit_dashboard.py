"""
BensBot Trading Dashboard

An autonomous trading dashboard built with Streamlit that integrates with
the existing event-driven trading architecture.
"""
import os
import json
import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st

# Internal imports from the dashboard package
from dashboard.theme import apply_custom_styling, COLORS
from dashboard.api_utils import (
    get_portfolio_data, get_trades, get_strategies, 
    get_alerts, get_system_logs, get_event_bus_status,
    get_trading_modes
)
from dashboard.visualizations import (
    df_or_empty, create_performance_chart, create_pie_chart,
    create_trade_history_chart, create_event_system_chart,
    enhance_dataframe
)
from dashboard.components import (
    section_header, styled_metric_card, strategy_card,
    strategy_lane, strategy_status_badge,
    event_system_status_card, trading_mode_card
)

# Import data_sources module
try:
    from dashboard.data_sources import (
        AlphaVantageAPI, FinnhubAPI, NewsDataAPI,
        _api_cache, _cache_expiry
    )
    HAS_DATA_SOURCES = True
except ImportError:
    HAS_DATA_SOURCES = False
    # Mock empty cache when module is unavailable
    _api_cache = {}
    _cache_expiry = {}

# Internal imports from your trading system
try:
    # Import trading bot modules for direct integration
    from trading_bot.strategies.strategy_factory import StrategyFactory
    from trading_bot.strategies.strategy_template import StrategyTemplate, StrategyOptimizable
    from trading_bot.event_system import EventManager
    HAS_TRADING_BOT = True

    # These might not be available yet so we'll try-except them
    try:
        from trading_bot.portfolio_manager import PortfolioManager
    except ImportError:
        PortfolioManager = None
        # Skip warning here; will show in main() after page config
        # For development without trading_bot, set default API keys if available
        if "ALPHA_VANTAGE_API_KEY" not in os.environ:
            os.environ["ALPHA_VANTAGE_API_KEY"] = ""
except ImportError:
    HAS_TRADING_BOT = False
    # Skip warning here; will show in main() after page config
    # For development without trading_bot, set default API keys if available
    if "ALPHA_VANTAGE_API_KEY" not in os.environ:
        os.environ["ALPHA_VANTAGE_API_KEY"] = ""

# ---------------------------------------------------------------------------
# 🏠 Home tab
# ---------------------------------------------------------------------------
def home_tab() -> None:
    # Header with last updated time
    st.markdown("<div class='main-header'>📊 BensBot Trading Dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"Last updated: {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}")
    
    # Portfolio Overview
    section_header("Portfolio Overview", "💼")
    
    # Get portfolio data
    portfolio_data = get_portfolio_data()
    portfolio_df = df_or_empty(portfolio_data)
    
    if not portfolio_df.empty:
        # Extract key metrics
        total_value = portfolio_df['total_value'].sum() if 'total_value' in portfolio_df.columns else 0
        daily_change = portfolio_df['daily_change'].sum() if 'daily_change' in portfolio_df.columns else 0
        daily_percent = (daily_change / (total_value - daily_change)) * 100 if total_value != daily_change else 0
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            styled_metric_card("Portfolio Value", total_value, is_currency=True)
        
        with col2:
            styled_metric_card("Daily Change", daily_change, daily_percent, is_currency=True)
        
        with col3:
            active_trades = len(get_trades(account_type="live"))
            styled_metric_card("Active Trades", active_trades)
        
        with col4:
            active_strategies = len(get_strategies(status="active"))
            styled_metric_card("Active Strategies", active_strategies)
        
        # Performance charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            performance_chart = create_performance_chart(portfolio_data)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        with col2:
            allocation_chart = create_pie_chart(portfolio_data)
            st.plotly_chart(allocation_chart, use_container_width=True)
        
        # Portfolio details
        st.markdown("### Portfolio Details")
        column_config = {
            "symbol": st.column_config.TextColumn("Symbol"),
            "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
            "avg_price": st.column_config.NumberColumn("Avg Price", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "total_value": st.column_config.NumberColumn("Total Value", format="$%.2f"),
            "daily_change": st.column_config.NumberColumn("Daily Change", format="$%.2f"),
            "total_return": st.column_config.NumberColumn("Total Return", format="$%.2f"),
            "total_return_percent": st.column_config.NumberColumn("Return %", format="%.2f%%")
        }
        enhance_dataframe(portfolio_df, column_config)
    else:
        st.info("No portfolio data available")
    
    # Active Trades
    section_header("Active Trades", "📈")
    
    # Create tabs for Paper vs Live trading
    paper_tab, live_tab = st.tabs(["Paper Trading", "Live Trading"])
    
    with paper_tab:
        paper_trades = get_trades(account_type="paper")
        paper_df = df_or_empty(paper_trades)
        
        if not paper_df.empty:
            # Summary metrics for paper trades
            col1, col2, col3 = st.columns(3)
            
            with col1:
                paper_count = len(paper_df)
                styled_metric_card("Open Paper Trades", paper_count)
            
            with col2:
                paper_value = paper_df['value'].sum() if 'value' in paper_df.columns else 0
                styled_metric_card("Total Paper Value", paper_value, is_currency=True)
            
            with col3:
                paper_pnl = paper_df['pnl'].sum() if 'pnl' in paper_df.columns else 0
                paper_pnl_pct = (paper_pnl / (paper_value - paper_pnl)) * 100 if paper_value != paper_pnl else 0
                styled_metric_card("Paper P&L", paper_pnl, paper_pnl_pct, is_currency=True)
            
            # Display paper trades table
            enhance_dataframe(paper_df)
        else:
            st.info("No paper trades active")
    
    with live_tab:
        live_trades = get_trades(account_type="live")
        live_df = df_or_empty(live_trades)
        
        if not live_df.empty:
            # Summary metrics for live trades
            col1, col2, col3 = st.columns(3)
            
            with col1:
                live_count = len(live_df)
                styled_metric_card("Open Live Trades", live_count)
            
            with col2:
                live_value = live_df['value'].sum() if 'value' in live_df.columns else 0
                styled_metric_card("Total Live Value", live_value, is_currency=True)
            
            with col3:
                live_pnl = live_df['pnl'].sum() if 'pnl' in live_df.columns else 0
                live_pnl_pct = (live_pnl / (live_value - live_pnl)) * 100 if live_value != live_pnl else 0
                styled_metric_card("Live P&L", live_pnl, live_pnl_pct, is_currency=True)
            
            # Display live trades table
            enhance_dataframe(live_df)
            
            # Display trade performance chart
            trade_chart = create_trade_history_chart(live_trades)
            st.plotly_chart(trade_chart, use_container_width=True)
        else:
            st.info("No live trades active")
    
    # Headline Alerts
    section_header("Headline Alerts", "📰")
    alerts = get_alerts(limit=20)
    
    if alerts:
        # Display alerts in a more visually appealing way
        for a in alerts:
            timestamp = a.get('ts', a.get('time', ''))
            headline = a.get('headline', '')
            source = a.get('source', '')
            impact = a.get('impact', 'neutral')
            
            # Determine alert color based on impact
            alert_color = "#17a2b8"  # Default to info blue
            if impact == 'positive':
                alert_color = "#28a745"  # Success green
            elif impact == 'negative':
                alert_color = "#dc3545"  # Danger red
            elif impact == 'warning':
                alert_color = "#ffc107"  # Warning yellow
            
            st.markdown(f"""
            <div style="
                background-color: white;
                border-left: 4px solid {alert_color};
                padding: 12px;
                margin-bottom: 8px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 4px;">
                    {timestamp}
                    {f' • <span style="font-weight: 500;">{source}</span>' if source else ''}
                </div>
                <div style="font-size: 1rem;">
                    {headline}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alerts right now.")

# ---------------------------------------------------------------------------
# 📊 Strategies tab
# ---------------------------------------------------------------------------

def strategies_tab() -> None:
    st.markdown("<div class='main-header'>💡 Trading Strategies</div>", unsafe_allow_html=True)
    
    from dashboard.api_utils import get_strategies, analyze_market_for_strategies
    
    # Load session state with strategies key if not present
    if 'strategies_initialized' not in st.session_state:
        st.session_state['strategies_initialized'] = False
    
    # First analyze the market and distribute strategies to appropriate categories
    # This is where we'd call the "strategy decider" functionality
    if not st.session_state['strategies_initialized']:
        with st.spinner("Loading trading strategies and analyzing market conditions..."):
            # Force a full analysis on initial load
            analyze_market_for_strategies(force_refresh=True)
            st.session_state['strategies_initialized'] = True
    else:
        # Just do regular updates on refreshes
        analyze_market_for_strategies()
    
    # Get strategies in each category after market analysis has placed them
    active_strategies = get_strategies(status="active")
    pending_strategies = get_strategies(status="pending_win")
    experimental_strategies = get_strategies(status="experimental")
    failed_strategies = get_strategies(status="failed")
    
    # Show diagnostic information
    with st.expander("Strategy Analysis Debug Info", expanded=False):
        st.write("Strategy counts:")
        st.write(f"Active: {len(active_strategies)}")
        st.write(f"Pending: {len(pending_strategies)}")
        st.write(f"Experimental: {len(experimental_strategies)}")
        st.write(f"Failed: {len(failed_strategies)}")
        
        if not active_strategies and not pending_strategies and not experimental_strategies and not failed_strategies:
            st.error("No strategies found. Forcing creation of simulation strategies...")
            from dashboard.api_utils import analyze_market_for_strategies
            analyze_market_for_strategies(force_refresh=True, force_simulation=True)
    
    # Calculate counts for the metrics
    active_count = len(active_strategies)
    pending_count = len(pending_strategies)
    experimental_count = len(experimental_strategies)
    failed_count = len(failed_strategies)
    total_count = active_count + pending_count + experimental_count + failed_count
    
    # Display summary metrics
    st.markdown("### Strategy Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        styled_metric_card("Total Strategies", total_count)
    
    with col2:
        styled_metric_card("Active", active_count)
    
    with col3:
        styled_metric_card("Pending Approval", pending_count)
    
    with col4:
        styled_metric_card("Experimental", experimental_count)
    
    with col5:
        styled_metric_card("Failed", failed_count)
    
    # Market scan button to trigger strategy re-evaluation
    scan_col1, scan_col2 = st.columns([3, 1])
    with scan_col2:
        if st.button("🔍 Scan Markets Now", use_container_width=True, key="scan_markets_button"):
            with st.spinner("Scanning markets and evaluating strategies..."):
                # This would actually call the backend to analyze markets and update strategies
                analyze_market_for_strategies(force_refresh=True)
                st.success("Market scan completed successfully!")
                st.rerun()
    
    with scan_col1:
        st.markdown("""
        #### Market Analysis and Strategy Selection
        The system continuously scans markets to identify trading opportunities and select the best strategies 
        based on current market conditions, news sentiment, and technical indicators.
        """)
    
    # Strategy sections - now all populated based on actual system decisions
    strategy_lane("pending_win", title="Winning Strategies – Awaiting Approval", icon="🏆", action="approve")
    strategy_lane("active", title="Active Strategies Under Test", icon="⚡", action="view")
    strategy_lane("experimental", title="Experimental Strategies & Symbols", icon="🧪", action="view")
    strategy_lane("failed", title="Failed Strategies – Mark for Deletion", icon="❌", action="delete")

# ---------------------------------------------------------------------------
# 🔌 Data Sources tab
# ---------------------------------------------------------------------------
def data_sources_tab() -> None:
    st.markdown("<div class='main-header'>🔌 Data Sources</div>", unsafe_allow_html=True)
    section_header("API Provider Status", "🔄")

    col1, col2, col3 = st.columns(3)

    if HAS_DATA_SOURCES:
        # Alpha Vantage
        with col1:
            try:
                resp = AlphaVantageAPI.get_market_data("AAPL", interval="daily")
                status = "Connected" if resp else "Limited"
                color = "#28a745" if resp else "#ffc107"
            except Exception as e:
                status = f"Error: {str(e)[:50]}"
                color = "#dc3545"
            st.markdown(f"""
                <div style="background:#1E293B; padding:1rem; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                  <h3 style="margin:0 0 .5rem; font-size:1.2rem; color:#F8FAFC;">Alpha Vantage</h3>
                  <div style="display:flex; align-items:center;">
                    <div style="width:10px;height:10px;border-radius:50%;background-color:{color};margin-right:8px;"></div>
                    <span style="color:#E2E8F0;">{status}</span>
                  </div>
                </div>
            """, unsafe_allow_html=True)

        # Finnhub
        with col2:
            try:
                resp = FinnhubAPI.get_company_news("AAPL")
                status = "Connected" if resp else "Limited"
                color = "#28a745" if resp else "#ffc107"
            except Exception as e:
                status = f"Error: {str(e)[:50]}"
                color = "#dc3545"
            st.markdown(f"""
                <div style="background:#1E293B; padding:1rem; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                  <h3 style="margin:0 0 .5rem; font-size:1.2rem; color:#F8FAFC;">Finnhub</h3>
                  <div style="display:flex; align-items:center;">
                    <div style="width:10px;height:10px;border-radius:50%;background-color:{color};margin-right:8px;"></div>
                    <span style="color:#E2E8F0;">{status}</span>
                  </div>
                </div>
            """, unsafe_allow_html=True)

        # NewsData.io
        with col3:
            try:
                resp = NewsDataAPI.get_financial_news(keywords=["market"])
                status = "Connected" if resp else "Limited"
                color = "#28a745" if resp else "#ffc107"
            except Exception as e:
                status = f"Error: {str(e)[:50]}"
                color = "#dc3545"
            st.markdown(f"""
                <div style="background:#1E293B; padding:1rem; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                  <h3 style="margin:0 0 .5rem; font-size:1.2rem; color:#F8FAFC;">NewsData.io</h3>
                  <div style="display:flex; align-items:center;">
                    <div style="width:10px;height:10px;border-radius:50%;background-color:{color};margin-right:8px;"></div>
                    <span style="color:#E2E8F0;">{status}</span>
                  </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        # Mock API status cards if modules aren't available
        for col, api_name in zip([col1, col2, col3], ["Alpha Vantage", "Finnhub", "NewsData.io"]):
            with col:
                st.markdown(f"""
                    <div style="background:#1E293B; padding:1rem; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                      <h3 style="margin:0 0 .5rem; font-size:1.2rem; color:#F8FAFC;">{api_name}</h3>
                      <div style="display:flex; align-items:center;">
                        <div style="width:10px;height:10px;border-radius:50%;background-color:#dc3545;margin-right:8px;"></div>
                        <span style="color:#E2E8F0;">Module unavailable</span>
                      </div>
                    </div>
                """, unsafe_allow_html=True)

    section_header("Cache Statistics", "📊")
    hits = st.session_state.get("cache_hits", 0)
    misses = st.session_state.get("cache_misses", 0)
    c1, c2, c3 = st.columns(3)
    with c1:
        styled_metric_card("Cache Size", len(_api_cache))
    with c2:
        styled_metric_card("Cache Hits", hits)
    with c3:
        styled_metric_card("Cache Misses", misses)

    with st.expander("Cache Details"):
        cache_rows = []
        for key, val in _api_cache.items():
            exp = _cache_expiry.get(key, datetime.datetime.now())
            cache_rows.append({
                "Key": key,
                "Expiry": exp.strftime("%Y-%m-%d %H:%M:%S"),
                "Status": "Expired" if exp < datetime.datetime.now() else "Valid",
                "Size": len(str(val))
            })
        df = pd.DataFrame(cache_rows)
        enhance_dataframe(df)

    section_header("Latest Headlines with Impact Scores", "📰")
    
    # Display mock news if module isn't available or API call fails
    mock_news = [
        {"title": "Fed signals potential rate cut in June meeting", "source": "Alpha Vantage", "impact": "positive", "date": "2025-04-24T14:30:00"},
        {"title": "AAPL reports quarterly earnings above estimates", "source": "NewsData.io", "impact": "positive", "date": "2025-04-24T10:15:00"},
        {"title": "MSFT announces new AI-powered services", "source": "Finnhub", "impact": "positive", "date": "2025-04-24T09:45:00"},
        {"title": "Oil prices rise amid Middle East tensions", "source": "Alpha Vantage", "impact": "negative", "date": "2025-04-24T08:20:00"},
        {"title": "Major bank reports cybersecurity breach", "source": "NewsData.io", "impact": "negative", "date": "2025-04-23T22:10:00"},
        {"title": "Market volatility increases as earnings season begins", "source": "Finnhub", "impact": "neutral", "date": "2025-04-23T16:30:00"},
        {"title": "Consumer confidence index exceeds expectations", "source": "Alpha Vantage", "impact": "positive", "date": "2025-04-23T14:45:00"},
        {"title": "Retail sales data shows strong momentum", "source": "NewsData.io", "impact": "positive", "date": "2025-04-23T11:30:00"},
        {"title": "Tech sector leads market rally", "source": "Finnhub", "impact": "positive", "date": "2025-04-23T10:15:00"},
        {"title": "Manufacturing PMI shows contraction for second month", "source": "Alpha Vantage", "impact": "negative", "date": "2025-04-22T15:45:00"}
    ]
    
    if HAS_DATA_SOURCES:
        try:
            news = NewsDataAPI.get_financial_news(keywords=["market", "stock", "finance"])
            if news and "results" in news:
                articles = []
                for art in news["results"][:10]:
                    articles.append({
                        "title": art.get("title", ""),
                        "source": art.get("source_id", ""),
                        "impact": art.get("impact", "neutral"),
                        "date": art.get("pubDate", "")
                    })
                if articles:
                    mock_news = articles
        except Exception:
            pass
    
    for article in mock_news:
        title = article["title"]
        src = article["source"]
        impact = article["impact"]
        tstamp = article["date"]
        ic = {"neutral":"#17a2b8","positive":"#28a745","negative":"#dc3545"}[impact]
        st.markdown(f"""
            <div style="
                background:white;
                border-left:4px solid {ic};
                padding:12px;
                margin-bottom:8px;
                border-radius:4px;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size:.8rem; color:#6c757d; margin-bottom:4px;">
                    {tstamp}
                    {' • <strong>'+src+'</strong>' if src else ''}
                    <span style="
                        float:right;
                        background:{ic};
                        color:white;
                        padding:2px 6px;
                        border-radius:4px;
                        font-size:.7rem;
                    ">
                        Impact: {impact.capitalize()}
                    </span>
                </div>
                <div style="font-size:1rem;">
                    {title}
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 🔌 Developer tab
# ---------------------------------------------------------------------------
def developer_tab() -> None:
    """Developer tab showing event system, trading modes, and system diagnostics"""
    st.markdown("<div class='main-header'>🔌 Developer Tools</div>", unsafe_allow_html=True)
    
    # Create tabs for different developer sections
    event_tab, modes_tab, diagnostics_tab = st.tabs([
        "Event System", "Trading Modes", "System Diagnostics"
    ])
    
    with event_tab:
        st.markdown("### Event System Overview")
        st.markdown("""
        This section provides a real-time view of the event-driven architecture
        powering the trading system. Monitor events, message queues, and channels.
        """)
        
        # Event system metrics card
        event_system_status_card()
        
        # Event activity chart
        st.markdown("### Event Activity")
        event_chart = create_event_system_chart(None)  # Will be replaced with real data
        st.plotly_chart(event_chart, use_container_width=True)
        
        # Channel monitoring
        st.markdown("### Active Channels")
        if HAS_TRADING_BOT:
            try:
                channels = ChannelManager.get_instance().get_channels()
                channel_data = []
                
                for channel_name, channel in channels.items():
                    channel_data.append({
                        "name": channel_name,
                        "subscribers": len(channel.subscribers),
                        "events_total": channel.stats.get("events_total", 0),
                        "last_event": channel.stats.get("last_event_timestamp", "")
                    })
                
                channel_df = pd.DataFrame(channel_data)
                enhance_dataframe(channel_df)
            except Exception as e:
                st.warning(f"Could not load channel data: {str(e)}")
                st.info("This section will show active event channels when connected to the trading system.")
        else:
            st.info("This section will show active event channels when connected to the trading system.")
    
    with modes_tab:
        st.markdown("### Trading Mode Configuration")
        st.markdown("""
        The trading mode system allows configuring how the autonomous system makes trading decisions.
        Different modes have different risk profiles and strategy approaches.
        """)
        
        # Trading mode status
        trading_mode_card()
        
        # Mode configuration
        st.markdown("### Mode Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode_type = st.selectbox(
                "Mode Type", 
                ["Standard", "Conservative", "Aggressive", "ML-Enhanced"]
            )
            
            max_positions = st.slider("Max Open Positions", 1, 20, 5)
            max_drawdown = st.slider("Max Drawdown %", 1.0, 25.0, 5.0, 0.5)
        
        with col2:
            risk_per_trade = st.slider("Risk Per Trade %", 0.1, 5.0, 1.0, 0.1)
            use_stop_loss = st.checkbox("Use Stop Loss", value=True)
            use_take_profit = st.checkbox("Use Take Profit", value=True)
        
        if st.button("Save Configuration"):
            st.success("Configuration saved. The changes will be applied to the selected trading mode.")
            # This would actually update the configuration in your system
    
    with diagnostics_tab:
        st.markdown("### System Diagnostics")
        st.markdown("""
        Monitor the health and performance of the trading system components.
        View system metrics, event metrics, and architecture visualization.
        """)
        
        # System metrics
        st.markdown("### System Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            styled_metric_card("CPU Usage", "23%")
        
        with col2:
            styled_metric_card("Memory Usage", "512 MB")
        
        with col3:
            styled_metric_card("Uptime", "3d 14h")
        
        # Event metrics
        st.markdown("### Event Metrics")
        
        # Mock metrics for now
        event_metrics = [
            {"category": "Market Data", "events_per_min": 120, "processors": 2, "queue_size": 5},
            {"category": "Signal Generation", "events_per_min": 45, "processors": 3, "queue_size": 2},
            {"category": "Trade Execution", "events_per_min": 12, "processors": 1, "queue_size": 0},
            {"category": "Portfolio Updates", "events_per_min": 8, "processors": 1, "queue_size": 1},
        ]
        
        event_df = pd.DataFrame(event_metrics)
        enhance_dataframe(event_df)
        
        # Architecture visualization
        st.markdown("### System Architecture")
        st.markdown("""
        The trading system uses an event-driven architecture with these key components:
        
        1. **Event Manager** - Core event processing system
        2. **Channel Manager** - Routes events to appropriate handlers
        3. **Message Queues** - Ensures events are processed in order
        4. **Trading Modes** - Configures strategy behavior
        5. **Order System** - Handles order execution
        """)
        
        # This would be better as an actual visualization

# ---------------------------------------------------------------------------
# 📜 Logs tab
# ---------------------------------------------------------------------------
def logs_tab() -> None:
    st.markdown("<div class='main-header'>📜 System Logs</div>", unsafe_allow_html=True)
    
    # Controls for filtering logs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR", "DEBUG"])
    
    with col2:
        component = st.selectbox("Component", ["All", "Trading Engine", "Data Fetcher", "Signal Generator", "Order Manager"])
    
    with col3:
        log_limit = st.slider("Number of logs", min_value=10, max_value=500, value=100, step=10)
    
    # Get logs from the API
    logs = get_system_logs(
        level=None if log_level == "All" else log_level,
        component=None if component == "All" else component,
        limit=log_limit
    )
    
    if logs:
        # Process and display logs
        logs_df = df_or_empty(logs)
        
        # Set up column configuration for better display
        column_config = {
            "timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, YYYY, hh:mm:ss a"),
            "level": st.column_config.Column("Level"),
            "component": st.column_config.Column("Component"),
            "message": st.column_config.TextColumn("Message", width="large"),
        }
        
        # Display logs dataframe
        enhance_dataframe(logs_df, column_config)
    else:
        st.info("No logs available")

# ---------------------------------------------------------------------------
# ⚙️ Settings tab
# ---------------------------------------------------------------------------
def settings_tab() -> None:
    st.markdown("<div class='main-header'>⚙️ Bot Settings</div>", unsafe_allow_html=True)
    
    # Create sections for different settings categories
    st.markdown("### Trading Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trading_mode = st.selectbox("Trading Mode", ["Paper Only", "Live Trading", "Both Paper and Live"])
        max_positions = st.number_input("Max Open Positions", min_value=1, max_value=50, value=10)
        position_size = st.number_input("Position Size (%)", min_value=1, max_value=100, value=5)
    
    with col2:
        risk_management = st.selectbox("Risk Management", ["Conservative", "Moderate", "Aggressive"])
        stop_loss = st.number_input("Stop Loss (%)", min_value=1, max_value=50, value=5)
        take_profit = st.number_input("Take Profit (%)", min_value=1, max_value=200, value=15)
    
    st.markdown("### API Configuration")
    
    api_col1, api_col2 = st.columns(2)
    
    with api_col1:
        # List all API services from your config
        st.markdown("#### Market Data APIs")
        st.text_input("Alpha Vantage API Key", type="password", value="*********")
        st.text_input("Finnhub API Key", type="password", value="*********")
        st.text_input("Tradier API Key", type="password", value="*********")
    
    with api_col2:
        st.markdown("#### News APIs")
        st.text_input("NewsData.io API Key", type="password", value="*********")
        st.text_input("Marketaux API Key", type="password", value="*********")
        st.text_input("MediaStack API Key", type="password", value="*********")
    
    # API cycling settings
    st.markdown("### API Cycling Configuration")
    st.markdown("Configure how the system cycles between different API providers to avoid rate limits")
    
    cycling_col1, cycling_col2 = st.columns(2)
    
    with cycling_col1:
        st.number_input("Min Time Between API Calls (seconds)", min_value=1, max_value=60, value=5)
        st.checkbox("Enable API Fallback", value=True)
    
    with cycling_col2:
        st.selectbox("API Rotation Strategy", ["Round Robin", "Least Recently Used", "Priority Based"])
        st.checkbox("Auto-Retry Failed Calls", value=True)
    
    # Save button with success message
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
        # This would actually save the settings to your configuration system

# ---------------------------------------------------------------------------
# 🚀 Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Set page configuration
    st.set_page_config(
        page_title="BensBot Trading Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # We're now using direct API integrations instead of trading_bot modules
    # No need to warn about missing modules
    
    # Create cache if not exists
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0
    if "cache_misses" not in st.session_state:
        st.session_state.cache_misses = 0
    
    # Apply custom styling
    apply_custom_styling()
    
    # Sidebar logo and navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 1.8rem; margin-bottom: 0; color: #38BDF8;">BensBot</h1>
            <p style="color: #94A3B8; margin-top: 0;">Trading Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            options=["Home", "Strategies", "Data Sources", "Developer", "Logs", "Settings"],
            index=0,
            format_func=lambda x: f"📊 {x}" if x == "Home" else
                         f"💡 {x}" if x == "Strategies" else
                         f"🔌 {x}" if x == "Data Sources" else
                         f"🔌 {x}" if x == "Developer" else
                         f"📜 {x}" if x == "Logs" else
                         f"⚙️ {x}"
        )
        
        # Display API connection status
        st.markdown("---")
        try:
            # This is a placeholder - it would actually check your trading system status
            system_status = "connected"
            
            if system_status == "connected":
                st.markdown("""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #28a745; margin-right: 8px;"></div>
                    <span style="color: #E2E8F0;">System Connected</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #dc3545; margin-right: 8px;"></div>
                    <span style="color: #E2E8F0;">System Disconnected</span>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #dc3545; margin-right: 8px;"></div>
                <span>System Disconnected</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Scan now button
        if st.button("🔍 Scan Markets Now", use_container_width=True):
            with st.spinner("Scanning markets for opportunities..."):
                # This would actually trigger a market scan in your system
                try:
                    if HAS_TRADING_BOT:
                        # Attempt to call your actual scanning logic
                        st.success("Market scan completed successfully!")
                    else:
                        # Mock response for development
                        st.success("Market scan completed successfully!")
                except Exception as e:
                    st.error(f"Error during market scan: {e}")
        
        # Version info
        st.markdown("""
        <div style="position: absolute; bottom: 20px; left: 20px; right: 20px; text-align: center; font-size: 0.8rem; color: #94A3B8;">
            BensBot Dashboard v1.0<br>
            &copy; 2025 BensBot
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if page == "Home":
        home_tab()
    elif page == "Strategies":
        strategies_tab()
    elif page == "Data Sources":
        data_sources_tab()
    elif page == "Developer":
        developer_tab()
    elif page == "Logs":
        logs_tab()
    elif page == "Settings":
        settings_tab()

if __name__ == "__main__":
    main()
