"""
UI Event Handler

Handles events from the trading system and updates the Streamlit UI.
Uses Streamlit's session state to store updates for rendering.
"""

import streamlit as st
import logging
import datetime
from trading_bot.core.event_system import EventType, Event, EventHandler

logger = logging.getLogger(__name__)

class UIEventHandler(EventHandler):
    """
    Handles events from the trading system and updates the Streamlit UI.
    Uses Streamlit's session state to store updates for rendering.
    """
    
    def __init__(self, event_bus):
        super().__init__(event_bus)
        logger.info("Initializing UI Event Handler")
        self._register_handlers()
        
    def _register_handlers(self):
        """Register handlers for different event types"""
        self.event_bus.add_handler(EventType.MARKET_DATA, self.handle_market_data)
        self.event_bus.add_handler(EventType.ORDER_UPDATE, self.handle_order_update)
        self.event_bus.add_handler(EventType.POSITION_UPDATE, self.handle_position_update)
        self.event_bus.add_handler(EventType.RISK_ALERT, self.handle_risk_alert)
        self.event_bus.add_handler(EventType.SIGNAL_GENERATED, self.handle_signal)
        self.event_bus.add_handler(EventType.PORTFOLIO_UPDATE, self.handle_portfolio_update)
        self.event_bus.add_handler(EventType.NEWS_UPDATE, self.handle_news_update)
        self.event_bus.add_handler(EventType.STRATEGY_UPDATE, self.handle_strategy_update)
        
        logger.info("UI Event Handler registered for all relevant events")
    
    def handle_market_data(self, event):
        """Handle market data updates"""
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
            
        symbol = event.data.get('symbol')
        if symbol:
            st.session_state.market_data[symbol] = event.data
            logger.debug(f"Updated market data for {symbol}")
    
    def handle_order_update(self, event):
        """Handle order updates"""
        if 'orders' not in st.session_state:
            st.session_state.orders = []
            
        # Add new order or update existing
        order_id = event.data.get('order_id')
        if order_id:
            # Check if order already exists
            existing = next((o for o in st.session_state.orders if o.get('order_id') == order_id), None)
            if existing:
                # Update existing order
                existing.update(event.data)
            else:
                # Add new order
                st.session_state.orders.append(event.data)
                
            logger.debug(f"Updated order {order_id} with status {event.data.get('status')}")
            
            # Add to alerts if status changed to filled or rejected
            if event.data.get('status') in ['filled', 'rejected', 'partially_filled']:
                self._add_alert(f"Order {order_id} {event.data.get('status')}", 
                               f"{event.data.get('side')} {event.data.get('quantity')} {event.data.get('symbol')} @ {event.data.get('price')}",
                               "success" if event.data.get('status') != 'rejected' else "error")
    
    def handle_position_update(self, event):
        """Handle position updates"""
        if 'positions' not in st.session_state:
            st.session_state.positions = {}
            
        symbol = event.data.get('symbol')
        if symbol:
            st.session_state.positions[symbol] = event.data
            logger.debug(f"Updated position for {symbol}")
    
    def handle_risk_alert(self, event):
        """Handle risk alerts"""
        level = event.data.get('level', 'warning')
        title = event.data.get('title', 'Risk Alert')
        message = event.data.get('message', '')
        
        self._add_alert(title, message, level)
        logger.info(f"Risk alert: {title} - {message}")
    
    def handle_signal(self, event):
        """Handle strategy signals"""
        if 'signals' not in st.session_state:
            st.session_state.signals = []
            
        st.session_state.signals.append(event.data)
        
        # Add signal to alerts
        strategy = event.data.get('strategy', 'Unknown')
        symbol = event.data.get('symbol', 'Unknown')
        action = event.data.get('action', 'Unknown')
        
        self._add_alert(
            f"{strategy} Signal", 
            f"{action} {symbol} at {event.data.get('price', 'market price')}",
            "success"
        )
        
        logger.info(f"Signal from {strategy}: {action} {symbol}")
    
    def handle_portfolio_update(self, event):
        """Handle portfolio updates"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
            
        # Update portfolio data
        st.session_state.portfolio = event.data
        logger.debug("Updated portfolio data")
    
    def handle_news_update(self, event):
        """Handle news updates"""
        if 'news' not in st.session_state:
            st.session_state.news = []
            
        # Add new news item
        news_item = event.data
        
        # Add timestamp if not present
        if 'timestamp' not in news_item:
            news_item['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Add to the front of the list to show newest first
        st.session_state.news.insert(0, news_item)
        
        # Keep only the latest 50 news items
        if len(st.session_state.news) > 50:
            st.session_state.news = st.session_state.news[:50]
            
        # Add to alerts for high impact news
        if news_item.get('impact', 'low') == 'high':
            self._add_alert(
                f"High Impact News: {news_item.get('title', 'News Update')}",
                news_item.get('content', ''),
                "warning"
            )
            
        logger.debug(f"Added news item: {news_item.get('title')}")
    
    def handle_strategy_update(self, event):
        """Handle strategy updates"""
        if 'strategies' not in st.session_state:
            st.session_state.strategies = {}
            
        strategy_id = event.data.get('id')
        if strategy_id:
            # Update strategy info
            st.session_state.strategies[strategy_id] = event.data
            
            # Add to alerts for important status changes
            if event.data.get('status_change'):
                self._add_alert(
                    f"Strategy Update: {event.data.get('name')}",
                    f"Status changed to {event.data.get('status')}",
                    "info"
                )
                
            logger.debug(f"Updated strategy {strategy_id}: {event.data.get('name')}")
    
    def _add_alert(self, title, message, level='info'):
        """Add an alert to the session state"""
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        alert = {
            "title": title,
            "message": message,
            "type": level,
            "time": current_time
        }
        
        # Add to the front of the list to show newest first
        st.session_state.alerts.insert(0, alert)
        
        # Keep only the latest 50 alerts
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts = st.session_state.alerts[:50]
            
        logger.info(f"Added alert [{level}]: {title} - {message}")
