#!/usr/bin/env python3
"""
Order Repository

This module provides the repository implementation for orders.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from trading_bot.persistence.mongo_repository import MongoRepository
from trading_bot.persistence.redis_repository import RedisRepository
from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.core.events import OrderAcknowledged, OrderFilled, OrderRejected, OrderCancelled


class OrderModel:
    """Data model for order persistence"""
    
    def __init__(
        self, 
        internal_id: str,
        broker: str, 
        symbol: str,
        quantity: float,
        side: str,
        order_type: str,
        time_in_force: str,
        status: str,
        broker_order_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        filled_quantity: float = 0.0,
        avg_fill_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _id: Optional[str] = None
    ):
        """
        Initialize an order model.
        
        Args:
            internal_id: Internal order ID
            broker: Broker identifier
            symbol: Asset symbol
            quantity: Order quantity
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop/etc)
            time_in_force: Time in force (day/gtc/etc)
            status: Order status (new/filled/canceled/rejected)
            broker_order_id: Broker-assigned order ID
            combo_id: ID of parent combo order if part of a multi-leg order
            limit_price: Limit price if applicable
            stop_price: Stop price if applicable
            created_at: Order creation timestamp
            updated_at: Last update timestamp
            filled_quantity: Quantity filled so far
            avg_fill_price: Average fill price
            metadata: Additional metadata
            _id: MongoDB document ID
        """
        self.internal_id = internal_id
        self.broker = broker
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.status = status
        self.broker_order_id = broker_order_id
        self.combo_id = combo_id
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.filled_quantity = filled_quantity
        self.avg_fill_price = avg_fill_price
        self.metadata = metadata or {}
        self._id = _id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderModel':
        """Create an OrderModel from a dictionary"""
        # Convert ObjectId to string if present
        if '_id' in data and not isinstance(data['_id'], str):
            data['_id'] = str(data['_id'])
            
        # Handle timestamp conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
            
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        result = {
            'internal_id': self.internal_id,
            'broker': self.broker,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'side': self.side,
            'order_type': self.order_type,
            'time_in_force': self.time_in_force,
            'status': self.status,
            'broker_order_id': self.broker_order_id,
            'combo_id': self.combo_id,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'metadata': self.metadata,
        }
        
        if self._id:
            result['_id'] = self._id
            
        return result
    
    @classmethod
    def from_order_acknowledged(cls, event: OrderAcknowledged) -> 'OrderModel':
        """Create an OrderModel from an OrderAcknowledged event"""
        return cls(
            internal_id=event.order_id,
            broker=event.broker,
            symbol=event.symbol,
            quantity=event.quantity,
            side=event.side,
            order_type=event.order_type,
            time_in_force=event.metadata.get('time_in_force', 'day') if event.metadata else 'day',
            status='acknowledged',
            broker_order_id=event.broker_order_id if hasattr(event, 'broker_order_id') else None,
            limit_price=event.limit_price,
            stop_price=event.metadata.get('stop_price') if event.metadata else None,
            metadata=event.metadata,
            created_at=event.timestamp if hasattr(event, 'timestamp') else datetime.now()
        )
    
    def update_from_fill(self, event: Union[OrderFilled, 'OrderPartialFill']) -> None:
        """Update order from a fill event"""
        self.updated_at = datetime.now()
        
        if isinstance(event, OrderFilled):
            self.status = 'filled'
            self.filled_quantity = event.total_qty
            self.avg_fill_price = event.avg_fill_price
        else:  # OrderPartialFill
            self.status = 'partially_filled'
            self.filled_quantity = self.quantity - event.remaining_qty
            
            # Update average fill price (weighted average)
            if self.avg_fill_price is None:
                self.avg_fill_price = event.fill_price
            else:
                # Calculate new weighted average price
                prev_fill_qty = self.filled_quantity - event.fill_qty
                prev_total = prev_fill_qty * self.avg_fill_price
                new_total = event.fill_qty * event.fill_price
                self.avg_fill_price = (prev_total + new_total) / self.filled_quantity


class OrderRepository:
    """Repository for Order persistence"""
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the order repository.
        
        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.mongo_repo = MongoRepository(connection_manager, 'orders')
        self.redis_repo = RedisRepository(connection_manager, 'orders')
        self.logger = logging.getLogger(__name__)
        
        # Idempotency cache
        self.id_cache = {}
    
    def save_order(self, order: Union[OrderModel, Dict[str, Any]]) -> str:
        """
        Save an order to the database.
        
        Args:
            order: Order model or dictionary to save
            
        Returns:
            Internal order ID
        """
        if not isinstance(order, OrderModel):
            order = OrderModel.from_dict(order)
            
        # Ensure we have an internal ID
        if not order.internal_id:
            order.internal_id = str(uuid.uuid4())
            
        # Update timestamp
        order.updated_at = datetime.now()
        
        try:
            # Save to MongoDB (durable storage)
            mongo_id = self.mongo_repo.save(order)
            
            # If MongoDB save succeeded, save to Redis (hot cache)
            try:
                self.redis_repo.save(order)
            except Exception as e:
                self.logger.warning(f"Failed to save order to Redis cache: {str(e)}")
            
            return order.internal_id
            
        except Exception as e:
            self.logger.error(f"Failed to save order: {str(e)}")
            raise
    
    def update_status(self, internal_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update order status.
        
        Args:
            internal_id: Internal order ID
            status: New status
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Find the order
            order = self.find_by_internal_id(internal_id)
            
            if not order:
                self.logger.warning(f"Cannot update status for unknown order: {internal_id}")
                return False
                
            # Update the order
            order.status = status
            order.updated_at = datetime.now()
            
            if metadata:
                if not order.metadata:
                    order.metadata = {}
                order.metadata.update(metadata)
            
            # Save to MongoDB
            if order._id:
                self.mongo_repo.update(order._id, order)
            else:
                self.mongo_repo.save(order)
            
            # Update in Redis
            try:
                self.redis_repo.update(internal_id, order)
            except Exception as e:
                self.logger.warning(f"Failed to update order in Redis cache: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update order status: {str(e)}")
            raise
    
    def update_from_event(self, event: Union[OrderAcknowledged, OrderFilled, OrderRejected, OrderCancelled, 'OrderPartialFill']) -> bool:
        """
        Update order from an event.
        
        Args:
            event: Order lifecycle event
            
        Returns:
            True if successful
        """
        try:
            internal_id = event.order_id
            
            # Handle acknowledged event
            if isinstance(event, OrderAcknowledged):
                # Check if order already exists
                order = self.find_by_internal_id(internal_id)
                
                if order:
                    # Order exists, update it
                    order.status = 'acknowledged'
                    order.broker_order_id = event.broker_order_id if hasattr(event, 'broker_order_id') else None
                    order.updated_at = datetime.now()
                    
                    # Save updates
                    if order._id:
                        self.mongo_repo.update(order._id, order)
                    else:
                        self.mongo_repo.save(order)
                else:
                    # Create new order
                    order = OrderModel.from_order_acknowledged(event)
                    self.mongo_repo.save(order)
                
                # Update Redis
                try:
                    self.redis_repo.update(internal_id, order)
                except Exception as e:
                    self.logger.warning(f"Failed to update order in Redis cache: {str(e)}")
                
                return True
                
            # Handle other events
            order = self.find_by_internal_id(internal_id)
            
            if not order:
                self.logger.warning(f"Cannot update unknown order from event: {internal_id}")
                return False
            
            # Update based on event type
            if isinstance(event, (OrderFilled, 'OrderPartialFill')):
                order.update_from_fill(event)
            elif isinstance(event, OrderCancelled):
                order.status = 'cancelled'
                order.updated_at = datetime.now()
            elif isinstance(event, OrderRejected):
                order.status = 'rejected'
                order.updated_at = datetime.now()
                if hasattr(event, 'reason') and event.reason:
                    if not order.metadata:
                        order.metadata = {}
                    order.metadata['rejection_reason'] = event.reason
            
            # Save to MongoDB
            if order._id:
                self.mongo_repo.update(order._id, order)
            else:
                self.mongo_repo.save(order)
            
            # Update in Redis
            try:
                self.redis_repo.update(internal_id, order)
            except Exception as e:
                self.logger.warning(f"Failed to update order in Redis cache: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update order from event: {str(e)}")
            raise
    
    def find_by_internal_id(self, internal_id: str) -> Optional[OrderModel]:
        """
        Find an order by internal ID.
        
        Args:
            internal_id: Internal order ID
            
        Returns:
            OrderModel if found, None otherwise
        """
        try:
            # Try Redis first
            try:
                order = self.redis_repo.find_by_id(internal_id)
                if order:
                    if isinstance(order, dict):
                        return OrderModel.from_dict(order)
                    return order
            except Exception:
                # Redis error, continue to MongoDB
                pass
            
            # Try MongoDB
            orders = self.mongo_repo.find_by_query({'internal_id': internal_id})
            
            if orders and len(orders) > 0:
                # Convert to OrderModel if needed
                if isinstance(orders[0], dict):
                    return OrderModel.from_dict(orders[0])
                return orders[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find order by internal ID: {str(e)}")
            raise
    
    def find_by_broker_order_id(self, broker: str, broker_order_id: str) -> Optional[OrderModel]:
        """
        Find an order by broker order ID.
        
        Args:
            broker: Broker identifier
            broker_order_id: Broker order ID
            
        Returns:
            OrderModel if found, None otherwise
        """
        try:
            # Try MongoDB
            orders = self.mongo_repo.find_by_query({
                'broker': broker,
                'broker_order_id': broker_order_id
            })
            
            if orders and len(orders) > 0:
                # Convert to OrderModel if needed
                if isinstance(orders[0], dict):
                    return OrderModel.from_dict(orders[0])
                return orders[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find order by broker order ID: {str(e)}")
            raise
    
    def find_by_combo_id(self, combo_id: str) -> List[OrderModel]:
        """
        Find all orders that are part of a combo.
        
        Args:
            combo_id: Combo order ID
            
        Returns:
            List of OrderModel
        """
        try:
            # Try MongoDB
            orders = self.mongo_repo.find_by_query({'combo_id': combo_id})
            
            # Convert to OrderModel if needed
            result = []
            for order in orders:
                if isinstance(order, dict):
                    result.append(OrderModel.from_dict(order))
                else:
                    result.append(order)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find orders by combo ID: {str(e)}")
            raise
    
    def fetch_open_orders(self) -> List[OrderModel]:
        """
        Fetch all open orders.
        
        Returns:
            List of OrderModel with open status
        """
        try:
            # Define open statuses
            open_statuses = ['new', 'acknowledged', 'partially_filled']
            
            # Try MongoDB
            orders = self.mongo_repo.find_by_query({
                'status': {'$in': open_statuses}
            })
            
            # Convert to OrderModel if needed
            result = []
            for order in orders:
                if isinstance(order, dict):
                    result.append(OrderModel.from_dict(order))
                else:
                    result.append(order)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders: {str(e)}")
            raise
    
    def get_idempotent_order_id(self, broker_order_id: str, broker: str) -> Optional[str]:
        """
        Get internal order ID for a broker order ID (for idempotency).
        
        Args:
            broker_order_id: Broker order ID
            broker: Broker identifier
            
        Returns:
            Internal order ID if exists
        """
        # Check cache first
        cache_key = f"{broker}:{broker_order_id}"
        if cache_key in self.id_cache:
            return self.id_cache[cache_key]
            
        # Check database
        order = self.find_by_broker_order_id(broker, broker_order_id)
        if order:
            # Cache the result
            self.id_cache[cache_key] = order.internal_id
            return order.internal_id
            
        return None
    
    def register_idempotent_mapping(self, internal_id: str, broker_order_id: str, broker: str) -> None:
        """
        Register internal ID to broker order ID mapping for idempotency.
        
        Args:
            internal_id: Internal order ID
            broker_order_id: Broker order ID
            broker: Broker identifier
        """
        # Update cache
        cache_key = f"{broker}:{broker_order_id}"
        self.id_cache[cache_key] = internal_id
