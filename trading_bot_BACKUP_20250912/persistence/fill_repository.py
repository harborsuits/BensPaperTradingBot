#!/usr/bin/env python3
"""
Fill Repository

This module provides the repository implementation for order fills.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from trading_bot.persistence.mongo_repository import MongoRepository
from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.core.events import OrderFilled, OrderPartialFill


class FillModel:
    """Data model for fill persistence"""
    
    def __init__(
        self,
        order_internal_id: str,
        fill_qty: float,
        fill_price: float,
        timestamp: datetime,
        event_type: str,  # 'partial_fill' or 'fill'
        fill_id: Optional[str] = None,
        symbol: Optional[str] = None,
        broker: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        commission: Optional[float] = None,
        fees: Optional[float] = None,
        side: Optional[str] = None,
        venue: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _id: Optional[str] = None
    ):
        """
        Initialize a fill model.
        
        Args:
            order_internal_id: Internal order ID
            fill_qty: Fill quantity
            fill_price: Fill price
            timestamp: Fill timestamp
            event_type: Event type ('partial_fill' or 'fill')
            fill_id: Unique fill identifier (generated if not provided)
            symbol: Asset symbol (optional)
            broker: Broker identifier (optional)
            broker_order_id: Broker order ID (optional)
            commission: Commission amount (optional)
            fees: Other fees (optional)
            side: Order side (buy/sell) (optional)
            venue: Execution venue (optional)
            metadata: Additional metadata
            _id: MongoDB document ID
        """
        self.order_internal_id = order_internal_id
        self.fill_qty = fill_qty
        self.fill_price = fill_price
        self.timestamp = timestamp
        self.event_type = event_type
        self.fill_id = fill_id or str(uuid.uuid4())
        self.symbol = symbol
        self.broker = broker
        self.broker_order_id = broker_order_id
        self.commission = commission
        self.fees = fees
        self.side = side
        self.venue = venue
        self.metadata = metadata or {}
        self._id = _id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FillModel':
        """Create a FillModel from a dictionary"""
        # Convert ObjectId to string if present
        if '_id' in data and not isinstance(data['_id'], str):
            data['_id'] = str(data['_id'])
            
        # Handle timestamp conversion
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        result = {
            'order_internal_id': self.order_internal_id,
            'fill_qty': self.fill_qty,
            'fill_price': self.fill_price,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'fill_id': self.fill_id,
            'symbol': self.symbol,
            'broker': self.broker,
            'broker_order_id': self.broker_order_id,
            'commission': self.commission,
            'fees': self.fees,
            'side': self.side,
            'venue': self.venue,
            'metadata': self.metadata
        }
        
        if self._id:
            result['_id'] = self._id
            
        return result
    
    @classmethod
    def from_order_partial_fill(cls, event: OrderPartialFill) -> 'FillModel':
        """Create a FillModel from an OrderPartialFill event"""
        return cls(
            order_internal_id=event.order_id,
            fill_qty=event.filled_qty,
            fill_price=event.fill_price,
            timestamp=event.timestamp if hasattr(event, 'timestamp') else datetime.now(),
            event_type='partial_fill',
            symbol=event.symbol,
            broker=event.broker,
            broker_order_id=event.broker_order_id if hasattr(event, 'broker_order_id') else None,
            side=event.side,
            metadata=event.metadata if hasattr(event, 'metadata') else None
        )
    
    @classmethod
    def from_order_filled(cls, event: OrderFilled) -> 'FillModel':
        """Create a FillModel from an OrderFilled event"""
        return cls(
            order_internal_id=event.order_id,
            fill_qty=event.total_qty,
            fill_price=event.avg_fill_price,
            timestamp=event.timestamp if hasattr(event, 'timestamp') else datetime.now(),
            event_type='fill',
            symbol=event.symbol,
            broker=event.broker,
            broker_order_id=event.broker_order_id if hasattr(event, 'broker_order_id') else None,
            side=event.side,
            metadata=event.metadata if hasattr(event, 'metadata') else None
        )


class FillRepository:
    """Repository for Fill persistence"""
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the fill repository.
        
        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.mongo_repo = MongoRepository(connection_manager, 'fills')
        self.logger = logging.getLogger(__name__)
    
    def record_fill(self, event: Union[OrderFilled, OrderPartialFill]) -> str:
        """
        Record a fill from an event.
        
        Args:
            event: Fill event
            
        Returns:
            Fill ID
        """
        try:
            # Convert event to fill model
            if isinstance(event, OrderFilled):
                fill = FillModel.from_order_filled(event)
            else:  # OrderPartialFill
                fill = FillModel.from_order_partial_fill(event)
                
            # Save to MongoDB
            self.mongo_repo.save(fill)
            
            return fill.fill_id
            
        except Exception as e:
            self.logger.error(f"Failed to record fill: {str(e)}")
            raise
    
    def find_by_order_id(self, order_internal_id: str) -> List[FillModel]:
        """
        Find fills by order ID.
        
        Args:
            order_internal_id: Internal order ID
            
        Returns:
            List of FillModel
        """
        try:
            # Get fills from MongoDB
            fills = self.mongo_repo.find_by_query({'order_internal_id': order_internal_id})
            
            # Convert to FillModel if needed
            result = []
            for fill in fills:
                if isinstance(fill, dict):
                    result.append(FillModel.from_dict(fill))
                else:
                    result.append(fill)
            
            # Sort by timestamp
            result.sort(key=lambda x: x.timestamp)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find fills by order ID: {str(e)}")
            raise
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[FillModel]:
        """
        Find fills within a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            List of FillModel
        """
        try:
            # Get fills from MongoDB
            fills = self.mongo_repo.find_by_query({
                'timestamp': {
                    '$gte': start_time.isoformat(),
                    '$lte': end_time.isoformat()
                }
            })
            
            # Convert to FillModel if needed
            result = []
            for fill in fills:
                if isinstance(fill, dict):
                    result.append(FillModel.from_dict(fill))
                else:
                    result.append(fill)
            
            # Sort by timestamp
            result.sort(key=lambda x: x.timestamp)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find fills by time range: {str(e)}")
            raise
    
    def find_by_symbol(self, symbol: str, start_time: Optional[datetime] = None) -> List[FillModel]:
        """
        Find fills for a symbol.
        
        Args:
            symbol: Asset symbol
            start_time: Optional start time filter
            
        Returns:
            List of FillModel
        """
        try:
            # Build query
            query = {'symbol': symbol}
            if start_time:
                query['timestamp'] = {'$gte': start_time.isoformat()}
            
            # Get fills from MongoDB
            fills = self.mongo_repo.find_by_query(query)
            
            # Convert to FillModel if needed
            result = []
            for fill in fills:
                if isinstance(fill, dict):
                    result.append(FillModel.from_dict(fill))
                else:
                    result.append(fill)
            
            # Sort by timestamp
            result.sort(key=lambda x: x.timestamp)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find fills by symbol: {str(e)}")
            raise
    
    def calculate_vwap(self, symbol: str, start_time: datetime, end_time: datetime) -> float:
        """
        Calculate Volume-Weighted Average Price for a symbol.
        
        Args:
            symbol: Asset symbol
            start_time: Start time
            end_time: End time
            
        Returns:
            VWAP or 0 if no fills
        """
        try:
            # Get fills
            fills = self.find_by_symbol(symbol)
            
            # Filter by time range
            fills = [f for f in fills if start_time <= f.timestamp <= end_time]
            
            if not fills:
                return 0.0
                
            # Calculate VWAP
            total_value = sum(f.fill_qty * f.fill_price for f in fills)
            total_volume = sum(f.fill_qty for f in fills)
            
            if total_volume == 0:
                return 0.0
                
            return total_value / total_volume
            
        except Exception as e:
            self.logger.error(f"Failed to calculate VWAP: {str(e)}")
            raise
