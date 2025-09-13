#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Module

This module provides the Order class and related enums for defining
trading orders and their properties.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """
    Enum representing the types of orders that can be placed in the trading system.
    
    Order types determine how and when orders are executed based on price conditions:
    - MARKET: Execute immediately at the best available price
    - LIMIT: Execute only at the specified price or better
    - STOP: Convert to a market order when the specified stop price is reached
    - STOP_LIMIT: Convert to a limit order when the specified stop price is reached
    - TRAILING_STOP: Dynamic stop order that adjusts with favorable price movements
    
    Each order type has specific use cases and risk profiles that make them suitable
    for different trading strategies and market conditions.
    """
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()

class OrderAction(Enum):
    """
    Enum representing the directional actions that can be taken when placing an order.
    
    Order actions define the direction of the trade and its effect on position:
    - BUY: Purchase a long position in the instrument
    - SELL: Sell an existing long position
    - BUY_TO_COVER: Purchase to close an existing short position
    - SELL_SHORT: Sell a borrowed instrument to open a short position
    
    These actions enable the full range of trading activities including
    establishing new positions, closing existing positions, and implementing
    more complex strategies like short selling.
    """
    BUY = auto()
    SELL = auto()
    BUY_TO_COVER = auto()  # For covering short positions
    SELL_SHORT = auto()    # For opening short positions

class OrderStatus(Enum):
    """
    Enum representing the lifecycle states of an order within the trading system.
    
    Order statuses track the complete lifecycle of an order from creation to final disposition:
    - CREATED: Initial state after order object creation but before submission
    - SUBMITTED: Order has been sent to the broker/exchange
    - ACCEPTED: Order has been validated and accepted by the broker/exchange
    - REJECTED: Order was not accepted due to validation errors or other issues
    - PARTIAL: Order has been partially filled but execution is not complete
    - FILLED: Order has been completely executed
    - CANCELLED: Order was canceled before completion
    - EXPIRED: Order reached its time limit without being filled
    - ERROR: An error occurred during order processing
    
    These statuses enable precise tracking of order progress and appropriate
    handling of various order outcomes throughout the trading system.
    """
    CREATED = auto()       # Just created, not submitted
    SUBMITTED = auto()     # Submitted to broker
    ACCEPTED = auto()      # Accepted by broker
    REJECTED = auto()      # Rejected by broker
    PARTIAL = auto()       # Partially filled
    FILLED = auto()        # Completely filled
    CANCELLED = auto()     # Cancelled before completion
    EXPIRED = auto()       # Expired without being filled
    ERROR = auto()         # Order encountered an error

class Order:
    """
    Comprehensive representation of a trading order within the system.
    
    The Order class encapsulates all information required to represent, track,
    and manage a trade throughout its complete lifecycle. It provides a rich 
    object model for order specifications, execution details, and state transitions
    that abstracts away broker-specific implementations.
    
    Key capabilities:
    1. Complete order specification for any instrument type
    2. Full order lifecycle state management
    3. Fill tracking with partial fill support
    4. Commission and fee accounting
    5. Trade grouping through trade IDs
    6. Serialization for persistence and communication
    
    The Order class serves as the core data structure for:
    - Representing trading intentions deterministically
    - Tracking execution progress
    - Calculating trade performance
    - Grouping related orders
    - Communicating with broker interfaces
    - Storing trade history
    
    Order instances maintain their complete history including timestamps,
    status transitions, and fill details, enabling comprehensive trade analysis
    and auditing capabilities.
    """
    
    def __init__(self, 
                symbol: str,
                order_type: OrderType,
                action: OrderAction,
                quantity: int,
                limit_price: Optional[float] = None,
                stop_price: Optional[float] = None,
                option_symbol: Optional[str] = None,
                trade_id: Optional[str] = None,
                order_id: Optional[str] = None,
                order_details: Optional[Dict[str, Any]] = None):
        """
        Initialize a new Order instance with specified parameters.
        
        Creates a new Order object with the given specification and automatically
        assigns a unique identifier if one is not provided. The order begins in the
        CREATED state and captures its creation timestamp for audit purposes.
        
        Parameters:
            symbol (str): The underlying asset symbol (e.g., "AAPL", "BTC-USD")
            order_type (OrderType): The type of order (MARKET, LIMIT, STOP, etc.)
            action (OrderAction): The action to take (BUY, SELL, BUY_TO_COVER, SELL_SHORT)
            quantity (int): Number of shares, contracts, or units to trade
            limit_price (Optional[float]): Price limit for LIMIT, STOP_LIMIT orders
            stop_price (Optional[float]): Trigger price for STOP, STOP_LIMIT orders
            option_symbol (Optional[str]): Option-specific symbol for options trading
                Format typically follows: "AAPL220121C00150000" (symbol, expiry, call/put, strike)
            trade_id (Optional[str]): Identifier to group related orders (e.g., legs of a spread)
            order_id (Optional[str]): Unique order identifier (auto-generated if None)
            order_details (Optional[Dict[str, Any]]): Additional order parameters:
                - 'time_in_force': Duration the order remains active ('DAY', 'GTC', 'IOC')
                - 'exchange': Specific exchange routing
                - 'strategy': Strategy identifier that generated this order
                - 'position_id': Related position identifier
                - 'custom_tags': User-defined tags for order categorization
                
        Notes:
            - Ensure required price parameters are provided for specific order types:
              * LIMIT orders require limit_price
              * STOP orders require stop_price
              * STOP_LIMIT orders require both stop_price and limit_price
            - The order starts in CREATED status and must be explicitly submitted
            - Timestamp is recorded automatically using the system clock
            - UUID generation ensures uniqueness across the trading system
            - Filled quantity starts at 0 and is updated as fills occur
            - Option orders should include both the underlying symbol and option-specific symbol
        """
        self.symbol = symbol
        self.order_type = order_type
        self.action = action
        self.quantity = quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.option_symbol = option_symbol
        self.trade_id = trade_id if trade_id else str(uuid.uuid4())
        self.order_id = order_id if order_id else str(uuid.uuid4())
        self.order_details = order_details or {}
        
        self.status = OrderStatus.CREATED
        self.creation_time = datetime.now()
        self.submission_time = None
        self.fill_time = None
        self.filled_quantity = 0
        self.average_fill_price = None
        self.commission = 0.0
        self.fees = 0.0
        
        logger.info(f"Created order: {self}")
    
    def __str__(self) -> str:
        """Return string representation of the order."""
        order_str = (
            f"Order(id={self.order_id}, "
            f"{self.action.name} {self.quantity} {self.symbol}"
        )
        
        if self.option_symbol:
            order_str += f" option {self.option_symbol}"
            
        order_str += f", type={self.order_type.name}"
        
        if self.limit_price is not None:
            order_str += f", limit=${self.limit_price:.2f}"
            
        if self.stop_price is not None:
            order_str += f", stop=${self.stop_price:.2f}"
            
        order_str += f", status={self.status.name})"
        
        return order_str
    
    def update_status(self, new_status: OrderStatus) -> None:
        """
        Update the order's status and trigger associated state changes.
        
        This method transitions the order to a new status state and performs
        any required side effects such as updating timestamps for significant
        status changes. Status updates are captured in the logs for audit purposes.
        
        Parameters:
            new_status (OrderStatus): The new status for the order
            
        Side effects:
            - Updates the order's status attribute
            - Records submission time when status changes to SUBMITTED
            - Records fill time when status changes to FILLED
            - Logs the status transition for audit purposes
            
        Status transition rules:
        - Most transitions are possible, but some logical constraints exist
        - Terminal statuses (FILLED, CANCELLED, REJECTED, ERROR, EXPIRED) 
          typically don't transition to other statuses
        - Proper transition sequencing should follow the logical order of 
          order processing (CREATED → SUBMITTED → ACCEPTED → PARTIAL → FILLED)
            
        Notes:
            - This method does not validate logical status transitions
            - For automated status updates from broker feeds, use this method
              to ensure consistent state changes
            - The order maintains its status history through timestamps
            - Status change logging provides an audit trail
        """
        old_status = self.status
        self.status = new_status
        
        if new_status == OrderStatus.SUBMITTED and self.submission_time is None:
            self.submission_time = datetime.now()
            
        if new_status == OrderStatus.FILLED and self.fill_time is None:
            self.fill_time = datetime.now()
        
        logger.info(f"Order {self.order_id} status updated: {old_status.name} -> {new_status.name}")
    
    def update_fill(self, filled_quantity: int, fill_price: float) -> None:
        """
        Update the order with fill information and recalculate order state.
        
        This method incorporates new fill information into the order's state
        by updating the filled quantity and calculating the volume-weighted
        average price across all fills. It also automatically updates the
        order's status based on the new fill quantity.
        
        Parameters:
            filled_quantity (int): Additional quantity filled in this update
            fill_price (float): Price at which this quantity was filled
            
        Side effects:
            - Increases the order's filled_quantity by the filled_quantity parameter
            - Calculates and updates the volume-weighted average fill price
            - Updates the order status to PARTIAL or FILLED as appropriate
            - Logs fill information for audit purposes
            
        Fill processing logic:
        1. Track the current filled quantity before the update
        2. Add the new fill quantity to the order's total filled quantity
        3. Calculate the new volume-weighted average price across all fills
        4. Update the order status based on completion percentage:
           - If 100% filled: set status to FILLED
           - If partially filled: set status to PARTIAL
            
        Notes:
            - This method should be called for each fill notification received
            - Multiple partial fills will be correctly tracked with proper VWAP
            - Fill time is automatically recorded on transition to FILLED status
            - The method performs no validation on the fill quantity or price
            - To record commission and fees, set those properties separately
        """
        # Update filled quantity
        old_filled = self.filled_quantity
        self.filled_quantity += filled_quantity
        
        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            # Weighted average of previous fills and new fill
            self.average_fill_price = (
                (old_filled * self.average_fill_price + filled_quantity * fill_price) /
                self.filled_quantity
            )
        
        # Update status based on fill
        if self.filled_quantity == self.quantity:
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIAL)
            
        logger.info(f"Order {self.order_id} filled: {filled_quantity} @ ${fill_price:.2f}, "
                  f"total filled: {self.filled_quantity}/{self.quantity}")
    
    def cancel(self) -> bool:
        """
        Attempt to cancel the order and update its status if cancellation is possible.
        
        This method attempts to transition the order to the CANCELLED status
        if it is in a state where cancellation is possible. Not all order states
        can be cancelled (e.g., already FILLED orders cannot be cancelled).
        
        Returns:
            bool: True if the order was successfully cancelled, False if the 
                  order cannot be cancelled due to its current status
                  
        Side effects:
            - If cancellable, updates the order status to CANCELLED
            - Logs the cancellation attempt result
            
        Cancellation rules:
        - Orders in CREATED, SUBMITTED, ACCEPTED, or PARTIAL status can be cancelled
        - Orders in terminal states (FILLED, CANCELLED, REJECTED, ERROR, EXPIRED) 
          cannot be cancelled
            
        Notes:
            - This method only updates the local order object status
            - To cancel an order with a broker, use the OrderManager.cancel_order() method
            - Cancellation may not be immediate with certain brokers or in high-volatility markets
            - Partially filled orders can be cancelled to prevent further fills
        """
        if self.status in (OrderStatus.CREATED, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, 
                          OrderStatus.PARTIAL):
            self.update_status(OrderStatus.CANCELLED)
            return True
        
        logger.warning(f"Cannot cancel order {self.order_id} with status {self.status.name}")
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the order to a dictionary representation.
        
        Creates a complete dictionary representation of the order containing all
        properties and state information, suitable for serialization, persistence,
        or transmission between system components.
        
        Returns:
            Dict[str, Any]: A dictionary containing all order properties and state,
                            with enum values converted to strings and datetime objects
                            converted to ISO-format strings
                            
        Dictionary structure:
        - Core order properties (symbol, type, action, quantity, prices)
        - Order identifiers (order_id, trade_id)
        - Current order state (status, filled quantity, average price)
        - Timestamps (creation, submission, fill)
        - Financial information (commission, fees)
        - Additional details (order_details dictionary)
        
        Notes:
            - All enum values are converted to their string names for serialization
            - All datetime objects are converted to ISO format strings
            - The resulting dictionary can be used with from_dict() to recreate the order
            - This format is suitable for JSON serialization
            - The dictionary representation maintains all information needed to
              fully reconstruct the order state
        """
        return {
            "order_id": self.order_id,
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "option_symbol": self.option_symbol,
            "order_type": self.order_type.name,
            "action": self.action.name,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.name,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "submission_time": self.submission_time.isoformat() if self.submission_time else None,
            "fill_time": self.fill_time.isoformat() if self.fill_time else None,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "fees": self.fees,
            "order_details": self.order_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Create an Order object from a dictionary representation.
        
        This class method reconstructs an Order object from a dictionary
        representation, typically created by the to_dict() method. It handles
        conversion from serialized string formats back to the appropriate
        enum values and datetime objects.
        
        Parameters:
            data (Dict[str, Any]): Dictionary representation of an order, 
                                  typically generated by to_dict()
            
        Returns:
            Order: A new Order instance with all properties restored from the dictionary
            
        Reconstruction process:
        1. Convert string representations of enums back to actual enum values
        2. Create a new Order instance with the basic parameters
        3. Restore the order's state (status, filled quantity, etc.)
        4. Convert ISO format strings back to datetime objects
        5. Restore financial details (commission, fees)
        
        Notes:
            - This method is the counterpart to to_dict() for serialization/deserialization
            - Used for reconstructing orders from persistent storage or message passing
            - Handles proper type conversions for all serialized properties
            - Returns a fully functional Order object with all state preserved
            - Primary use cases include loading saved orders and reconstructing 
              orders from broker notifications
        """
        # Convert string enum values to actual enum values
        order_type = getattr(OrderType, data['order_type'])
        action = getattr(OrderAction, data['action'])
        
        # Create order with basic parameters
        order = cls(
            symbol=data['symbol'],
            order_type=order_type,
            action=action,
            quantity=data['quantity'],
            limit_price=data.get('limit_price'),
            stop_price=data.get('stop_price'),
            option_symbol=data.get('option_symbol'),
            trade_id=data.get('trade_id'),
            order_id=data.get('order_id'),
            order_details=data.get('order_details', {})
        )
        
        # Update status
        if 'status' in data:
            order.status = getattr(OrderStatus, data['status'])
        
        # Update timestamps
        if 'creation_time' in data and data['creation_time']:
            order.creation_time = datetime.fromisoformat(data['creation_time'])
        
        if 'submission_time' in data and data['submission_time']:
            order.submission_time = datetime.fromisoformat(data['submission_time'])
        
        if 'fill_time' in data and data['fill_time']:
            order.fill_time = datetime.fromisoformat(data['fill_time'])
        
        # Update fill information
        order.filled_quantity = data.get('filled_quantity', 0)
        order.average_fill_price = data.get('average_fill_price')
        order.commission = data.get('commission', 0.0)
        order.fees = data.get('fees', 0.0)
        
        return order 