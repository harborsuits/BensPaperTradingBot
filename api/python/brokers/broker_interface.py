"""
Broker Interface

This module defines the abstract interface for broker integrations, allowing
the trading system to work with multiple brokers and implement failover.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, time, timedelta
import logging

from trading_bot.core.events import OrderAcknowledged, OrderPartialFill, OrderFilled, OrderCancelled, OrderRejected, SlippageMetric
from trading_bot.event_system.event_bus import EventBus

logger = logging.getLogger(__name__)

class BrokerInterface(ABC):
    """
    Abstract interface for broker implementations
    
    All broker adapters must implement these methods to ensure
    compatibility with the trade executor.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        self.broker_id = 'unknown'  # To be set by implementing classes

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        pass
    
    @abstractmethod
    def get_next_market_open(self) -> datetime:
        """
        Get the next market open datetime
        
        Returns:
            datetime: Next market open time
        """
        pass
    
    @abstractmethod
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get trading hours information
        
        Returns:
            Dict: Trading hours information for the current day
        """
        pass
    
    @abstractmethod
    def get_account_balances(self) -> Dict[str, Any]:
        """
        Get account balance information
        
        Returns:
            Dict: Account balance details
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
            List[Dict]: List of positions
        """
        pass
    
    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current orders
        
        Returns:
            List[Dict]: List of orders
        """
        pass
    
    @abstractmethod
    def place_equity_order(self, 
                          symbol: str, 
                          quantity: int, 
                          side: str, 
                          order_type: str, 
                          time_in_force: str = 'day', 
                          limit_price: float = None, 
                          stop_price: float = None, 
                          expected_price: float = None) -> Dict[str, Any]:
        """
        Place an equity order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Order duration ('day', 'gtc')
            limit_price: Limit price (required for limit and stop_limit orders)
            duration: Order duration ('day', 'gtc')
            price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            
        Returns:
            Dict: Order result with ID and status
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Order status details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Cancellation result
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Quote information
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, 
                           symbol: str, 
                           interval: str, 
                           start_date: datetime, 
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get historical market data
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date
            end_date: End date (defaults to current time)
            
        Returns:
            List[Dict]: List of OHLCV candles
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get broker name
        
        Returns:
            str: Broker name
        """
        pass
    
    @property
    @abstractmethod
    def status(self) -> str:
        """
        Get broker connection status
        
        Returns:
            str: Status ('connected', 'disconnected', 'error')
        """
        pass
    
    @property
    @abstractmethod
    def supports_extended_hours(self) -> bool:
        """
        Check if broker supports extended hours trading
        
        Returns:
            bool: True if extended hours trading is supported
        """
        pass
    
    @property
    @abstractmethod
    def supports_fractional_shares(self) -> bool:
        """
        Check if broker supports fractional shares
        
        Returns:
            bool: True if fractional shares are supported
        """
        pass
    
    @property
    def api_calls_remaining(self) -> Optional[int]:
        """
        Get number of API calls remaining (rate limiting)
        
        Returns:
            Optional[int]: Number of calls remaining or None if not applicable
        """
        return None
    
    @abstractmethod
    def get_broker_time(self) -> datetime:
        """
        Get current time from broker's servers
        
        Returns:
            datetime: Current time according to broker
        """
        pass
    
    def needs_refresh(self) -> bool:
        """
        Check if broker connection needs refresh
        
        Returns:
            bool: True if connection needs to be refreshed
        """
        return False
    
    @abstractmethod
    def refresh_connection(self) -> bool:
        """
        Refresh broker connection (re-authenticate)
        
        Returns:
            bool: True if refresh successful
        """
        pass
        
    @abstractmethod
    def get_margin_status(self) -> Dict[str, Any]:
        """
        Get margin account status
        
        Returns:
            Dict containing:
                account_id (str): Account identifier
                cash (float): Available cash
                margin_used (float): Current borrowed amount
                buying_power (float): Max purchasing capacity
                maintenance_requirement (float): Maintenance margin level
        """
        pass


class MarketSession:
    """
    Market session management utilities
    
    This class handles market hours and session information for better
    market-aware trading decisions.
    """
    
    # Regular market hours (Eastern Time)
    REGULAR_OPEN = time(9, 30)
    REGULAR_CLOSE = time(16, 0)
    
    # Pre-market hours (Eastern Time)
    PREMARKET_OPEN = time(4, 0)
    
    # After-hours (Eastern Time)
    AFTERHOURS_CLOSE = time(20, 0)
    
    # Trading days (0 = Monday, 6 = Sunday)
    TRADING_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday
    
    @staticmethod
    def is_trading_day(date: datetime) -> bool:
        """
        Check if date is a trading day (weekday)
        
        Args:
            date: Date to check
            
        Returns:
            bool: True if trading day
        """
        return date.weekday() in MarketSession.TRADING_DAYS
    
    @staticmethod
    def is_regular_hours(dt: datetime) -> bool:
        """
        Check if time is during regular market hours
        
        Args:
            dt: Datetime to check
            
        Returns:
            bool: True if during regular hours
        """
        return (
            MarketSession.is_trading_day(dt) and
            MarketSession.REGULAR_OPEN <= dt.time() < MarketSession.REGULAR_CLOSE
        )
    
    @staticmethod
    def is_extended_hours(dt: datetime) -> bool:
        """
        Check if time is during extended market hours
        
        Args:
            dt: Datetime to check
            
        Returns:
            bool: True if during extended hours
        """
        if not MarketSession.is_trading_day(dt):
            return False
            
        return (
            (MarketSession.PREMARKET_OPEN <= dt.time() < MarketSession.REGULAR_OPEN) or
            (MarketSession.REGULAR_CLOSE <= dt.time() < MarketSession.AFTERHOURS_CLOSE)
        )
    
    @staticmethod
    def get_next_market_open(dt: datetime) -> datetime:
        """
        Get next market open time from given datetime
        
        Args:
            dt: Current datetime
            
        Returns:
            datetime: Next market open time
        """
        current_date = dt.date()
        
        # If it's before market open today
        if dt.time() < MarketSession.REGULAR_OPEN and MarketSession.is_trading_day(dt):
            return datetime.combine(current_date, MarketSession.REGULAR_OPEN)
        
        # Otherwise, find the next trading day
        days_ahead = 1
        while days_ahead < 8:  # Look up to a week ahead
            next_date = current_date + timedelta(days=days_ahead)
            next_dt = datetime.combine(next_date, MarketSession.REGULAR_OPEN)
            
            if MarketSession.is_trading_day(next_dt):
                return next_dt
                
            days_ahead += 1
            
        # Fallback - shouldn't reach here unless calendar is broken
        return dt + timedelta(days=3)  # Just go 3 days ahead
    
    @staticmethod
    def get_market_close_today(dt: datetime) -> Optional[datetime]:
        """
        Get market close time for today
        
        Args:
            dt: Current datetime
            
        Returns:
            Optional[datetime]: Market close time or None if not a trading day
        """
        if not MarketSession.is_trading_day(dt):
            return None
            
        current_date = dt.date()
        return datetime.combine(current_date, MarketSession.REGULAR_CLOSE)
    
    @staticmethod
    def get_session_type(dt: datetime) -> str:
        """
        Get current market session type
        
        Args:
            dt: Current datetime
            
        Returns:
            str: Session type ('regular', 'pre', 'post', 'closed')
        """
        if not MarketSession.is_trading_day(dt):
            return "closed"
            
        time_of_day = dt.time()
        
        if MarketSession.REGULAR_OPEN <= time_of_day < MarketSession.REGULAR_CLOSE:
            return "regular"
        elif MarketSession.PREMARKET_OPEN <= time_of_day < MarketSession.REGULAR_OPEN:
            return "pre"
        elif MarketSession.REGULAR_CLOSE <= time_of_day < MarketSession.AFTERHOURS_CLOSE:
            return "post"
        else:
            return "closed"
    
    @staticmethod
    def get_time_to_next_session(dt: datetime) -> timedelta:
        """
        Get time until next session starts
        
        Args:
            dt: Current datetime
            
        Returns:
            timedelta: Time until next session
        """
        session = MarketSession.get_session_type(dt)
        
        if session == "regular":
            # Time until close
            close_today = MarketSession.get_market_close_today(dt)
            return close_today - dt
        elif session == "pre":
            # Time until regular open
            open_today = datetime.combine(dt.date(), MarketSession.REGULAR_OPEN)
            return open_today - dt
        elif session == "post":
            # Time until next day's pre-market
            next_open = MarketSession.get_next_market_open(dt)
            pre_open = datetime.combine(next_open.date(), MarketSession.PREMARKET_OPEN)
            return pre_open - dt
        else:
            # Closed, time until next pre-market
            next_open = MarketSession.get_next_market_open(dt)
            pre_open = datetime.combine(next_open.date(), MarketSession.PREMARKET_OPEN)
            return pre_open - dt
