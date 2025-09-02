#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Order Manager Module

This module provides the OrderManager class for managing the lifecycle
of orders in the trading system.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from trading_bot.orders.order import Order, OrderStatus, OrderType, OrderAction

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Comprehensive order lifecycle management system for the trading platform.
    
    The OrderManager serves as the central coordinator for all order-related operations,
    providing a complete order management solution that bridges strategy signals with 
    broker execution while maintaining a robust audit trail of all trading activities.
    
    Key responsibilities:
    1. Order submission and routing to appropriate brokers
    2. Order cancellation and modification handling
    3. Order status tracking and state management
    4. Fill processing and position impact calculation
    5. Trade grouping and relationship management
    6. Order history and analytics for performance measurement
    7. Persistence and recovery of order state
    
    This component acts as the crucial middleware between:
    - Strategy components generating trading signals
    - Broker connectors executing orders in the market
    - Position management tracking overall exposure
    - Risk management enforcing trading constraints
    - Performance analysis measuring execution quality
    - Compliance systems maintaining a complete audit trail
    
    Implementation considerations:
    - Thread safety for concurrent order operations
    - Idempotent processing of broker callbacks
    - Comprehensive error handling for broker communication failures
    - Efficient indexing for order retrieval by various attributes
    - Serialization support for persistence and recovery
    - Simulation mode for backtesting and development
    
    The OrderManager maintains complete order history and relationships,
    enabling sophisticated order analytics, trade grouping, and performance
    measurement across the entire trading system.
    """
    
    def __init__(self, broker_connector=None):
        """
        Initialize the OrderManager with an optional broker connection.
        
        Creates a new OrderManager instance that can operate in either live
        trading mode (with a broker connector) or simulation mode (without a
        connector). All internal tracking structures are initialized empty.
        
        Parameters:
            broker_connector: Connection to broker API for order execution.
                If provided, enables live trading with the specified broker.
                If None, operates in simulation mode with mock execution.
                
        The broker_connector should implement these methods:
        - submit_order(order): Send order to broker, return success boolean
        - cancel_order(order_id): Cancel order with broker, return success boolean
        - modify_order(order_id, modifications): Modify existing order, return success boolean
        - get_order_status(order_id): Query current status of an order
        
        Notes:
            - In simulation mode, orders are automatically marked as filled
            - The OrderManager can be initialized without a broker and connected later
            - Multiple OrderManager instances can be created for different brokers
            - Thread safety should be considered when sharing instances
            - Internal state is empty until orders are submitted
        """
        self.broker_connector = broker_connector
        self.orders: Dict[str, Order] = {}  # Order ID to Order mapping
        self.trades: Dict[str, List[Order]] = {}  # Trade ID to list of Orders
        
        logger.info("Initialized OrderManager")
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution and begin tracking it.
        
        This method represents the entry point for all new orders in the system.
        It registers the order in the tracking collections and attempts to submit
        it to the broker if one is connected. The order's status is updated
        based on the submission result.
        
        Parameters:
            order (Order): The order to submit for execution. Must be a properly
                initialized Order object with all required fields for the order type.
                
        Returns:
            bool: True if the order was successfully submitted (or simulated in
                 simulation mode), False if submission failed
                 
        Order submission process:
        1. Register the order in internal tracking collections
        2. Group the order with other orders in the same trade (if any)
        3. Submit to broker if connected, otherwise simulate
        4. Update order status based on submission result
        
        Side effects:
            - Adds the order to the internal orders dictionary
            - Adds the order to the appropriate trade group
            - Updates the order's status based on submission result
            - In simulation mode, automatically fills the order
            
        Error handling:
            - Broker connection errors are caught and logged
            - Order status is set to ERROR if submission fails
            - Returns False if submission fails for any reason
            
        Notes:
            - Order registration happens before submission to ensure tracking
            - In live mode, the order status is set to SUBMITTED upon successful submission
            - In simulation mode, the order status is set directly to FILLED
            - Order validation should be performed before calling this method
            - Use the order's trade_id to group related orders (e.g., spread legs)
        """
        # Store order in our tracking collections
        self.orders[order.order_id] = order
        
        if order.trade_id not in self.trades:
            self.trades[order.trade_id] = []
        self.trades[order.trade_id].append(order)
        
        # If we have a broker connection, submit the order
        if self.broker_connector:
            try:
                success = self.broker_connector.submit_order(order)
                if success:
                    order.update_status(OrderStatus.SUBMITTED)
                else:
                    order.update_status(OrderStatus.ERROR)
                return success
            except Exception as e:
                logger.error(f"Error submitting order: {e}")
                order.update_status(OrderStatus.ERROR)
                return False
        else:
            # Mock submission (simulation mode)
            logger.info(f"Simulating order submission: {order}")
            order.update_status(OrderStatus.SUBMITTED)
            # Immediately set to FILLED in simulation mode for simplicity
            order.update_status(OrderStatus.FILLED)
            order.update_fill(order.quantity, order.limit_price or 0.0)
            return True
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order through the broker or simulation.
        
        This method attempts to cancel an order identified by its order_id.
        It validates that the order exists and is in a state where cancellation
        is possible, then submits the cancellation request to the broker if
        connected or simulates cancellation in simulation mode.
        
        Parameters:
            order_id (str): Unique identifier of the order to cancel
            
        Returns:
            bool: True if cancellation was successful, False if the order cannot
                  be cancelled or cancellation failed
                  
        Cancellation process:
        1. Verify the order exists in the tracking system
        2. If connected to a broker, submit cancellation request
        3. If in simulation mode, attempt to cancel the order locally
        4. Update order status if cancellation succeeds
        
        Side effects:
            - Updates the order's status to CANCELLED if successful
            - Logs the cancellation attempt and result
            
        Error handling:
            - Returns False if the order doesn't exist
            - Returns False if the order is in a state that cannot be cancelled
            - Catches and logs broker communication errors
            
        Notes:
            - Some order states cannot be cancelled (e.g., FILLED orders)
            - Cancellation is only attempted if the order exists in the tracking system
            - In live trading, the broker may reject cancellation requests
            - Order status is only updated if cancellation is confirmed
            - Cancellation may not be immediate, especially in fast markets
        """
        if order_id not in self.orders:
            logger.warning(f"Cannot cancel unknown order ID: {order_id}")
            return False
            
        order = self.orders[order_id]
        
        # If we have a broker connection, cancel the order
        if self.broker_connector:
            try:
                success = self.broker_connector.cancel_order(order_id)
                if success:
                    order.cancel()
                return success
            except Exception as e:
                logger.error(f"Error cancelling order: {e}")
                return False
        else:
            # Mock cancellation
            logger.info(f"Simulating order cancellation: {order}")
            return order.cancel()
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Retrieve an order by its unique identifier.
        
        Provides direct access to an order object based on its order_id,
        enabling inspection of order details, status, and execution information.
        
        Parameters:
            order_id (str): The unique identifier of the order to retrieve
            
        Returns:
            Optional[Order]: The requested Order object if found, None if no order
                           exists with the specified ID
                           
        Use cases:
            - Retrieving order details for display or analysis
            - Checking order status during strategy execution
            - Accessing execution details for filled orders
            - Extracting order parameters for modification
            
        Notes:
            - This method only returns orders tracked by this OrderManager instance
            - Returns None if the order_id is not found
            - Returns a reference to the actual Order object, not a copy
            - Changes to the returned object will affect the tracked order
            - For refreshing order status from the broker, use a separate method
        """
        return self.orders.get(order_id)
    
    def get_orders_by_trade(self, trade_id: str) -> List[Order]:
        """
        Retrieve all orders associated with a specific trade.
        
        This method returns all orders that share the same trade_id, which
        is used to group related orders such as the legs of a multi-leg trade
        or entry/exit pairs for a position.
        
        Parameters:
            trade_id (str): The trade identifier to retrieve orders for
            
        Returns:
            List[Order]: A list of all orders associated with the trade_id.
                        Returns an empty list if no orders are found.
                        
        Trade grouping use cases:
            - Retrieving all legs of a spread or multi-leg option strategy
            - Tracking entry and exit orders for a single position
            - Grouping related orders for reporting and analysis
            - Associating parent and child orders in bracket order structures
            
        Notes:
            - Orders are grouped by trade_id at submission time
            - The returned list may contain orders in various states
            - Returns an empty list if the trade_id is not found
            - Returns references to the actual Order objects, not copies
            - Orders are returned in the order they were submitted
            - No guarantee of execution sequence is provided
        """
        return self.trades.get(trade_id, [])
    
    def get_all_orders(self) -> List[Order]:
        """
        Retrieve all orders tracked by this OrderManager.
        
        Returns a complete list of all orders in the system, regardless of
        their status or relationship. This provides access to the full order
        history for analysis, reporting, or bulk operations.
        
        Returns:
            List[Order]: A list of all Order objects currently tracked
            
        Use cases:
            - Generating comprehensive trading reports
            - Performing analysis across all orders
            - Exporting full order history
            - Debugging and system verification
            
        Performance considerations:
            - This method returns references to all tracked orders
            - For large order histories, consider using filtered queries
            - Consider using get_orders_by_status() for active order monitoring
            
        Notes:
            - Orders are returned in no particular order
            - For ordered results, sort the returned list by creation_time or other attributes
            - Returns references to the actual Order objects, not copies
            - The returned list may be empty if no orders have been submitted
            - This method returns all orders for the lifetime of this OrderManager instance
        """
        return list(self.orders.values())
    
    def get_active_orders(self) -> List[Order]:
        """
        Retrieve all orders that are currently active in the market.
        
        This method returns all orders that have been submitted but have not
        yet reached a terminal state (FILLED, CANCELLED, REJECTED, ERROR, or EXPIRED).
        These are orders that may still execute or require monitoring.
        
        Returns:
            List[Order]: A list of all orders in non-terminal states
            
        Active order states include:
            - CREATED: Created but not yet submitted
            - SUBMITTED: Sent to broker but not confirmed
            - ACCEPTED: Confirmed by broker but not executed
            - PARTIAL: Partially filled but not complete
            
        Use cases:
            - Monitoring currently active orders
            - Risk management and exposure calculation
            - Order book reconciliation with broker
            - Cleanup of stale orders
            
        Notes:
            - This is a convenience method equivalent to filtering by active statuses
            - Orders are returned in no particular order
            - Returns an empty list if no active orders exist
            - Critical for monitoring orders that still require attention
            - Useful for implementing order timeout management
        """
        active_statuses = {
            OrderStatus.CREATED,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL
        }
        
        return [order for order in self.orders.values() if order.status in active_statuses]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Retrieve all orders with a specific status.
        
        This method filters the tracked orders to return only those with the
        specified status, enabling targeted monitoring and management of
        orders based on their current state.
        
        Parameters:
            status (OrderStatus): The status to filter orders by
            
        Returns:
            List[Order]: A list of Order objects with the specified status
            
        Common status queries:
            - FILLED: For processing completed trades
            - ERROR: For investigating and resolving issues
            - PARTIAL: For monitoring partially filled orders
            - SUBMITTED: For tracking orders awaiting confirmation
            
        Use cases:
            - Monitoring all orders in a specific state
            - Processing filled orders for position updates
            - Investigating rejected or error orders
            - Cleaning up expired or cancelled orders
            
        Notes:
            - Returns an empty list if no orders have the specified status
            - Orders are returned in no particular order
            - For chronological order, sort the results by creation_time
            - This method performs filtering on the client side
            - Returns references to the actual Order objects, not copies
        """
        return [order for order in self.orders.values() if order.status == status]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        Retrieve all orders for a specific underlying symbol.
        
        This method returns all orders related to a particular underlying
        instrument, regardless of their status or type. It enables focused
        analysis and management of trading activity for a specific asset.
        
        Parameters:
            symbol (str): The underlying symbol to filter orders by (e.g., "AAPL")
            
        Returns:
            List[Order]: A list of all orders for the specified symbol
            
        Use cases:
            - Monitoring all activity for a specific asset
            - Calculating total exposure to a particular symbol
            - Analyzing trading history for a specific instrument
            - Implementing symbol-specific risk controls
            
        Notes:
            - Matches against the underlying symbol, not option symbols
            - Returns orders in any status (active or completed)
            - Returns an empty list if no orders match the symbol
            - For option orders, matches the underlying symbol
            - For chronological order, sort by creation_time
            - Returns references to the actual Order objects, not copies
        """
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        """
        Update the status of an existing order.
        
        This method changes the status of an order identified by its order_id.
        It is primarily used to reflect status changes reported by the broker,
        such as when an order is accepted, rejected, or otherwise changes state.
        
        Parameters:
            order_id (str): The unique identifier of the order to update
            status (OrderStatus): The new status to set for the order
            
        Returns:
            bool: True if the order was found and updated, False otherwise
            
        Status update workflow:
        1. Locate the order in the tracking system
        2. If found, update its status using the Order.update_status() method
        3. Return the result of the operation
        
        Side effects:
            - Updates the order's status if found
            - May trigger associated timestamp updates in the Order object
            - Logs the status update for audit purposes
            
        Notes:
            - Returns False if the order_id is not found
            - This method is typically called in response to broker callbacks
            - No validation of valid status transitions is performed
            - The Order.update_status() method handles status-specific side effects
            - Status updates are logged for audit trail purposes
            - For terminal statuses, consider additional processing requirements
        """
        if order_id not in self.orders:
            logger.warning(f"Cannot update status of unknown order ID: {order_id}")
            return False
            
        order = self.orders[order_id]
        order.update_status(status)
        return True
    
    def update_order_fill(self, order_id: str, filled_quantity: int, 
                        fill_price: float) -> bool:
        """
        Update an order with fill information.
        
        This method records execution details for an order, updating its
        filled quantity and average price. It handles both complete and
        partial fills, and is typically called in response to execution
        reports from the broker.
        
        Parameters:
            order_id (str): The unique identifier of the order being filled
            filled_quantity (int): The quantity filled in this execution
            fill_price (float): The price at which the order was filled
            
        Returns:
            bool: True if the order was found and updated, False otherwise
            
        Fill processing workflow:
        1. Locate the order in the tracking system
        2. If found, update its fill information using Order.update_fill()
        3. Return the result of the operation
        
        Side effects:
            - Updates the order's filled quantity and average fill price
            - May update order status based on fill completeness
            - Logs the fill details for audit purposes
            
        Notes:
            - Returns False if the order_id is not found
            - This method is typically called in response to fill notifications
            - The Order.update_fill() method handles fill-related side effects
            - Partial fills update status to PARTIAL
            - Complete fills update status to FILLED
            - Multiple partial fills are correctly tracked with proper VWAP calculation
            - Fill information is critical for position tracking and P&L calculation
        """
        if order_id not in self.orders:
            logger.warning(f"Cannot update fill of unknown order ID: {order_id}")
            return False
            
        order = self.orders[order_id]
        order.update_fill(filled_quantity, fill_price)
        return True
    
    def to_json(self) -> str:
        """
        Serialize the OrderManager's state to a JSON string.
        
        This method creates a complete JSON representation of the OrderManager's
        current state, including all tracked orders and their relationships.
        The resulting JSON can be used for persistence, transmission, or
        state reconstruction.
        
        Returns:
            str: A JSON string containing the complete serialized state
            
        Serialized state includes:
        - Complete dictionary of all tracked orders
        - Trade grouping relationships
        - All order details and status information
        
        Use cases:
            - Persisting order state to disk for recovery
            - Transmitting order state between system components
            - Creating execution reports or audit trails
            - System state backup and restoration
            
        Notes:
            - JSON format enables easy storage and interoperability
            - Order objects are serialized using their to_dict() method
            - The entire order history is included in the output
            - The resulting JSON can be used with from_json() for reconstruction
            - May generate large outputs for systems with many orders
            - Indentation is included for human readability
        """
        state = {
            "orders": {order_id: order.to_dict() for order_id, order in self.orders.items()},
            "trades": {trade_id: [order.order_id for order in orders] 
                      for trade_id, orders in self.trades.items()}
        }
        
        return json.dumps(state, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str, order_factory: Callable = Order.from_dict) -> 'OrderManager':
        """
        Reconstruct an OrderManager instance from a JSON string.
        
        This class method creates a new OrderManager and restores its state
        from a JSON string, typically one created by the to_json() method.
        It rebuilds the complete order tracking state, including all orders
        and their relationships.
        
        Parameters:
            json_str (str): JSON string containing the serialized state
            order_factory (Callable): Function to create Order objects from dictionaries,
                                      defaults to Order.from_dict
            
        Returns:
            OrderManager: A new OrderManager instance with state restored from the JSON
            
        Reconstruction process:
        1. Create a new OrderManager instance
        2. Parse the JSON string to extract state information
        3. Recreate all orders using the provided order factory function
        4. Rebuild the trade grouping relationships
        5. Return the fully reconstructed OrderManager
        
        Use cases:
            - System state recovery after restart
            - Loading saved order history for analysis
            - Transferring order state between system components
            - Testing and simulation setup
            
        Notes:
            - The order factory parameter allows for custom Order reconstruction
            - Order objects are created using the specified factory function
            - Trade relationships are preserved in the reconstruction
            - The broker connector is not restored and must be set separately
            - Any orders not referenced in trade groups will still be tracked
            - Validates order references to ensure consistency
        """
        manager = cls()
        state = json.loads(json_str)
        
        # Reconstruct orders
        for order_id, order_data in state.get("orders", {}).items():
            order = order_factory(order_data)
            manager.orders[order_id] = order
        
        # Reconstruct trades
        for trade_id, order_ids in state.get("trades", {}).items():
            manager.trades[trade_id] = [manager.orders.get(order_id) for order_id in order_ids 
                                      if order_id in manager.orders]
        
        return manager 