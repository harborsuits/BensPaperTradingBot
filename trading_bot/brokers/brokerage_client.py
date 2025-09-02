"""
Brokerage Client Interface

This module defines the base interface for broker API clients and implements
specific broker integrations (Alpaca, etc.). It also provides connection 
monitoring to ensure stable communication with broker APIs.
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Standardized order types across brokers"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    
class OrderSide(Enum):
    """Standardized order sides across brokers"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"
    
class TimeInForce(Enum):
    """Standardized time-in-force values across brokers"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    
class BrokerConnectionStatus(Enum):
    """Connection status for broker APIs"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"  # Connected but with issues
    RECONNECTING = "reconnecting"
    AUTHENTICATING = "authenticating"

class BrokerAPIError(Exception):
    """Base exception for broker API errors"""
    def __init__(self, message: str, broker: str, error_code: Optional[str] = None, 
                response: Optional[Dict] = None):
        self.message = message
        self.broker = broker
        self.error_code = error_code
        self.response = response
        super().__init__(f"{broker} API Error: {message}" + 
                        (f" (Code: {error_code})" if error_code else ""))

class BrokerAuthError(BrokerAPIError):
    """Authentication error with broker API"""
    pass

class BrokerConnectionError(BrokerAPIError):
    """Connection error with broker API"""
    pass

class OrderExecutionError(BrokerAPIError):
    """Error executing an order"""
    pass

class BrokerageClient(ABC):
    """
    Abstract base class for broker API clients.
    
    This class defines the standard interface that all broker
    implementations should follow. It provides methods for
    authentication, account data, market data, and order execution.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the broker client.
        
        Args:
            config_path: Path to configuration file (optional)
            **kwargs: Additional configuration options
        """
        self.config = {}
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize connection monitoring
        self.connection_status = BrokerConnectionStatus.DISCONNECTED
        self.last_connection_check = None
        self.connection_errors = []
        self.max_connection_errors = 3
        self.reconnect_delay = 5  # seconds
        
        # Initialize order selection logic
        self.available_order_types = set()
        self.available_time_in_force = set()
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the broker API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker API.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def check_connection(self) -> BrokerConnectionStatus:
        """
        Check the current connection status.
        
        Returns:
            BrokerConnectionStatus: Current connection status
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances.
        
        Returns:
            Dict[str, Any]: Account information
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            status: Filter by order status (optional)
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        pass
    
    @abstractmethod
    def place_order(self, 
                  symbol: str, 
                  side: Union[OrderSide, str], 
                  quantity: float, 
                  order_type: Union[OrderType, str] = OrderType.MARKET,
                  time_in_force: Union[TimeInForce, str] = TimeInForce.DAY,
                  limit_price: Optional[float] = None, 
                  stop_price: Optional[float] = None,
                  trail_price: Optional[float] = None,
                  trail_percent: Optional[float] = None,
                  client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (buy, sell, sell_short, buy_to_cover)
            quantity: Order quantity
            order_type: Type of order (market, limit, stop, etc.)
            time_in_force: Time in force for the order (day, gtc, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            trail_price: Trailing amount for trailing stop orders
            trail_percent: Trailing percentage for trailing stop orders
            client_order_id: Client-specified order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict[str, Any]: Cancellation status
        """
        pass
    
    @abstractmethod
    def modify_order(self, 
                   order_id: str,
                   quantity: Optional[float] = None,
                   order_type: Optional[Union[OrderType, str]] = None,
                   time_in_force: Optional[Union[TimeInForce, str]] = None,
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            order_type: New order type
            time_in_force: New time in force
            limit_price: New limit price
            stop_price: New stop price
            
        Returns:
            Dict[str, Any]: Modified order information
        """
        pass
    
    @abstractmethod
    def get_market_hours(self, market: str = "equity") -> Dict[str, Any]:
        """
        Get market hours information.
        
        Args:
            market: Market to get hours for (equity, options, etc.)
            
        Returns:
            Dict[str, Any]: Market hours information
        """
        pass
    
    @abstractmethod
    def is_market_open(self, market: str = "equity") -> bool:
        """
        Check if a market is currently open.
        
        Args:
            market: Market to check (equity, options, etc.)
            
        Returns:
            bool: True if market is open, False otherwise
        """
        pass
    
    def get_supported_order_types(self) -> List[OrderType]:
        """
        Get order types supported by this broker.
        
        Returns:
            List[OrderType]: List of supported order types
        """
        return list(self.available_order_types)
    
    def get_supported_time_in_force(self) -> List[TimeInForce]:
        """
        Get time-in-force options supported by this broker.
        
        Returns:
            List[TimeInForce]: List of supported time in force options
        """
        return list(self.available_time_in_force)
    
    def _handle_connection_error(self, error: Exception) -> None:
        """
        Handle a connection error.
        
        Args:
            error: The exception that occurred
        """
        # Record the error
        self.connection_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "type": type(error).__name__
        })
        
        # Trim error list to keep only recent errors
        if len(self.connection_errors) > self.max_connection_errors:
            self.connection_errors = self.connection_errors[-self.max_connection_errors:]
        
        # Update connection status
        if len(self.connection_errors) >= self.max_connection_errors:
            self.connection_status = BrokerConnectionStatus.DISCONNECTED
        else:
            self.connection_status = BrokerConnectionStatus.DEGRADED
        
        # Log the error
        logger.error(f"Broker connection error: {str(error)}")
    
    def _select_order_type(self, 
                         desired_type: Union[OrderType, str], 
                         has_limit_price: bool, 
                         has_stop_price: bool) -> OrderType:
        """
        Select the appropriate order type based on available types and parameters.
        
        Args:
            desired_type: Desired order type
            has_limit_price: Whether a limit price is provided
            has_stop_price: Whether a stop price is provided
            
        Returns:
            OrderType: Selected order type
        """
        # Convert string to enum if needed
        if isinstance(desired_type, str):
            try:
                desired_type = OrderType(desired_type)
            except ValueError:
                logger.warning(f"Unknown order type: {desired_type}, using MARKET instead")
                desired_type = OrderType.MARKET
        
        # If desired type is available, use it
        if desired_type in self.available_order_types:
            return desired_type
        
        # Otherwise, select an appropriate alternative
        if desired_type == OrderType.STOP_LIMIT and has_limit_price and has_stop_price:
            # Try to find a suitable alternative
            if OrderType.STOP in self.available_order_types:
                logger.warning("STOP_LIMIT not available, using STOP instead")
                return OrderType.STOP
            elif OrderType.LIMIT in self.available_order_types:
                logger.warning("STOP_LIMIT not available, using LIMIT instead")
                return OrderType.LIMIT
        
        if desired_type == OrderType.STOP and has_stop_price:
            if OrderType.STOP_LIMIT in self.available_order_types and has_limit_price:
                logger.warning("STOP not available, using STOP_LIMIT instead")
                return OrderType.STOP_LIMIT
        
        if desired_type == OrderType.LIMIT and has_limit_price:
            if OrderType.STOP_LIMIT in self.available_order_types and has_stop_price:
                logger.warning("LIMIT not available, using STOP_LIMIT instead")
                return OrderType.STOP_LIMIT
        
        # Default to MARKET if all else fails
        logger.warning(f"{desired_type.value} not available, using MARKET instead")
        return OrderType.MARKET
    
    def _select_time_in_force(self, desired_tif: Union[TimeInForce, str]) -> TimeInForce:
        """
        Select the appropriate time-in-force based on available options.
        
        Args:
            desired_tif: Desired time in force
            
        Returns:
            TimeInForce: Selected time in force
        """
        # Convert string to enum if needed
        if isinstance(desired_tif, str):
            try:
                desired_tif = TimeInForce(desired_tif)
            except ValueError:
                logger.warning(f"Unknown time in force: {desired_tif}, using DAY instead")
                desired_tif = TimeInForce.DAY
        
        # If desired option is available, use it
        if desired_tif in self.available_time_in_force:
            return desired_tif
        
        # Otherwise, select an appropriate alternative
        if desired_tif == TimeInForce.GTC:
            logger.warning("GTC not available, using DAY instead")
        elif desired_tif == TimeInForce.IOC:
            logger.warning("IOC not available, using DAY instead")
        elif desired_tif == TimeInForce.FOK:
            logger.warning("FOK not available, using DAY instead")
        
        # Default to DAY if all else fails
        return TimeInForce.DAY

# Dictionary of broker class implementations, populated by imports
BROKER_IMPLEMENTATIONS = {} 