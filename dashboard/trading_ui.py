"""
Trading UI - Main Application

This is the main entry point for the enhanced trading dashboard UI.
It connects to the trading engine and event system while providing
a beautiful and functional user interface.
"""

import streamlit as st
import os
import sys
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from plotly.subplots import make_subplots

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_ui")

# Import UI modules
from dashboard.ui_styles import (
    ThemeMode, UIColors, UITypography, UIEffects, UISpacing,
    apply_base_styles, format_currency, format_percentage,
    create_card, create_metric_card, theme_plotly_chart
)

# Import trading system components
try:
    from trading_bot.core.event_system import EventBus, Event, EventType
    from trading_bot.core.event_handlers.ui_event_handler import UIEventHandler
    from trading_bot.core.order_manager import OrderManager
    from trading_bot.core.position_manager import PositionManager
    from trading_bot.core.broker.multi_broker_manager import MultiBrokerManager
    from trading_bot.core.strategy_manager import StrategyManager
    from trading_bot.strategies_new.strategy_factory import StrategyFactory
    from trading_bot.core.algo_execution import TWAPExecutor, VWAPExecutor, IcebergExecutor
    from trading_bot.core.cross_asset_risk_manager import CrossAssetRiskManager
    from trading_bot.ai.minerva.multi_ai_coordinator import MultiAICoordinator
    
    TRADING_ENGINE_AVAILABLE = True
    logger.info("Successfully imported trading engine components")
except ImportError as e:
    logger.warning(f"Could not import trading engine components: {e}")
    logger.warning("Running in demo mode with simulated data")
    TRADING_ENGINE_AVAILABLE = False

# Import dashboard components
from dashboard.components.dashboard_tab import render_dashboard_tab
from dashboard.components.backtester_tab import render_backtester_tab
from dashboard.components.news_tab import render_news_tab
from dashboard.components.developer_tab import render_developer_tab

# Page configuration
st.set_page_config(
    page_title="BensBot Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize event system and trading components
def initialize_trading_system():
    """Initialize the trading system components and connect to the event bus"""
    if not TRADING_ENGINE_AVAILABLE:
        logger.warning("Trading engine components not available. Using simulated data.")
        return None, None, None, None, None, None, None, None
    
    try:
        # Initialize event bus
        event_bus = EventBus()
        logger.info("Event bus initialized")
        
        # Initialize UI event handler
        ui_handler = UIEventHandler(event_bus)
        logger.info("UI event handler initialized")
        
        # Initialize trading components
        order_manager = OrderManager(event_bus)
        position_manager = PositionManager(event_bus)
        broker_manager = MultiBrokerManager(event_bus)
        risk_manager = CrossAssetRiskManager(event_bus)
        strategy_manager = StrategyManager(event_bus)
        strategy_factory = StrategyFactory()
        
        # Initialize AI coordinator
        ai_coordinator = MultiAICoordinator()
        
        logger.info("All trading components initialized successfully")
        
        return event_bus, order_manager, position_manager, broker_manager, risk_manager, strategy_manager, strategy_factory, ai_coordinator
    except Exception as e:
        logger.error(f"Error initializing trading system: {e}")
        return None, None, None, None, None, None, None, None

# Initialize the trading system at startup
event_bus, order_manager, position_manager, broker_manager, risk_manager, strategy_manager, strategy_factory, ai_coordinator = initialize_trading_system()

# Initialize session state for first run
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = ThemeMode.DARK

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Dashboard"

if 'notifications' not in st.session_state:
    st.session_state.notifications = []

if 'trading_mode' not in st.session_state:
    st.session_state.trading_mode = "Paper Trading"

# Apply the selected theme
apply_base_styles(st.session_state.theme_mode)

def toggle_theme():
    """Toggle between dark and light themes"""
    if st.session_state.theme_mode == ThemeMode.DARK:
        st.session_state.theme_mode = ThemeMode.LIGHT
    else:
        st.session_state.theme_mode = ThemeMode.DARK
    apply_base_styles(st.session_state.theme_mode)
    st.rerun()

def set_active_tab(tab_name):
    """Set the active tab and rerun the app"""
    st.session_state.active_tab = tab_name
    st.rerun()

def switch_trading_mode(new_mode):
    """Switch between trading modes"""
    if not TRADING_ENGINE_AVAILABLE:
        st.warning("Trading engine not available. Cannot switch modes.")
        return
    
    try:
        # Save previous mode
        previous_mode = st.session_state.trading_mode
        
        # Update session state
        st.session_state.trading_mode = new_mode
        
        # Emit event to notify system of mode change
        if event_bus:
            event_bus.emit(Event(
                EventType.SYSTEM_MODE_CHANGE,
                {
                    "previous_mode": previous_mode,
                    "new_mode": new_mode,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ))
        
        logger.info(f"Trading mode switched from {previous_mode} to {new_mode}")
        
        # Show success message
        st.success(f"Successfully switched to {new_mode} mode")
    except Exception as e:
        logger.error(f"Error switching trading mode: {e}")
        st.error(f"Failed to switch trading mode: {str(e)}")

def toggle_strategy(strategy_name, enabled):
    """Enable or disable a strategy"""
    if not TRADING_ENGINE_AVAILABLE or not strategy_manager:
        return
    
    try:
        # Get strategy instance
        strategy = strategy_manager.get_strategy_by_name(strategy_name)
        
        if strategy:
            if enabled:
                strategy_manager.enable_strategy(strategy.id)
                logger.info(f"Enabled strategy: {strategy_name}")
            else:
                strategy_manager.disable_strategy(strategy.id)
                logger.info(f"Disabled strategy: {strategy_name}")
            
            # Emit event for UI update
            if event_bus:
                event_bus.emit(Event(
                    EventType.STRATEGY_UPDATE,
                    {
                        "id": strategy.id,
                        "name": strategy_name,
                        "enabled": enabled,
                        "status": "active" if enabled else "inactive",
                        "status_change": True
                    }
                ))
    except Exception as e:
        logger.error(f"Error toggling strategy {strategy_name}: {e}")

def get_portfolio_summary():
    """Get portfolio summary from position manager or use demo data"""
    if TRADING_ENGINE_AVAILABLE and broker_manager and position_manager:
        try:
            # Get actual portfolio data
            portfolio = st.session_state.get('portfolio', {})
            
            if not portfolio:
                # Try to get data from broker manager
                account = broker_manager.get_account_summary()
                
                if account:
                    portfolio_value = account.get('portfolio_value', 0)
                    previous_value = account.get('previous_value', 0)
                    daily_change = portfolio_value - previous_value
                    daily_percent = (daily_change / previous_value * 100) if previous_value else 0
                    
                    return portfolio_value, daily_change, daily_percent
            else:
                portfolio_value = portfolio.get('total_value', 0)
                previous_value = portfolio.get('previous_value', 0)
                daily_change = portfolio_value - previous_value
                daily_percent = (daily_change / previous_value * 100) if previous_value else 0
                
                return portfolio_value, daily_change, daily_percent
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
    
    # Demo data
    portfolio_value = 125750.43
    daily_change = 1245.67
    daily_percent = (daily_change / (portfolio_value - daily_change)) * 100
    
    return portfolio_value, daily_change, daily_percent

def get_active_strategies():
    """Get active strategies from strategy manager or use demo data"""
    if TRADING_ENGINE_AVAILABLE and strategy_manager:
        try:
            strategies = {}
            
            for strategy in strategy_manager.get_all_strategies():
                strategies[strategy.name] = strategy.enabled
            
            if strategies:
                return strategies
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
    
    # Demo data
    return {
        "Gap Trading": True, 
        "Trend Following": True,
        "News Sentiment": False,
        "Volume Surge": True,
    }

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x80?text=BensBot", width=150)
        st.title("BensBot Trading")
        
        # Portfolio summary in sidebar
        portfolio_value, daily_change, daily_percent = get_portfolio_summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}", 
                     f"{daily_change:+,.2f} ({daily_percent:+.2f}%)",
                     delta_color="normal")
        
        # Trading mode indicator
        st.caption(f"Mode: {st.session_state.trading_mode}")
        
        # Main navigation
        st.subheader("Navigation")
        
        # Use styled buttons for navigation
        tabs = ["Dashboard", "Backtester", "News/Predictions", "Developer"]
        
        for tab in tabs:
            button_style = "primary" if st.session_state.active_tab == tab else "secondary"
            if st.button(tab, key=f"nav_{tab}", use_container_width=True, type=button_style):
                set_active_tab(tab)
        
        # Additional sidebar components
        st.divider()
        
        # Quick access strategy toggles
        st.subheader("Active Strategies")
        strategies = get_active_strategies()
        
        for strategy, enabled in strategies.items():
            if st.checkbox(strategy, value=enabled, key=f"strategy_{strategy}"):
                if not enabled:  # State changed from disabled to enabled
                    toggle_strategy(strategy, True)
            else:
                if enabled:  # State changed from enabled to disabled
                    toggle_strategy(strategy, False)
        
        st.divider()
        
        # Theme toggle in sidebar
        theme_label = "Switch to Light Theme" if st.session_state.theme_mode == ThemeMode.DARK else "Switch to Dark Theme"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)
        
        # Display current time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last updated: {current_time}")
    
    # Main content area based on active tab
    if st.session_state.active_tab == "Dashboard":
        render_dashboard_tab(
            position_manager=position_manager, 
            broker_manager=broker_manager,
            risk_manager=risk_manager,
            ai_coordinator=ai_coordinator
        )
    elif st.session_state.active_tab == "Backtester":
        render_backtester_tab(
            strategy_manager=strategy_manager,
            strategy_factory=strategy_factory
        )
    elif st.session_state.active_tab == "News/Predictions":
        render_news_tab(
            ai_coordinator=ai_coordinator
        )
    elif st.session_state.active_tab == "Developer":
        render_developer_tab(
            event_bus=event_bus,
            order_manager=order_manager,
            position_manager=position_manager,
            broker_manager=broker_manager,
            risk_manager=risk_manager,
            strategy_manager=strategy_manager
        )

if __name__ == "__main__":
    main()
