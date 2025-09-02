#!/usr/bin/env python3
"""
Position Repository

This module provides the repository implementation for positions.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set

from trading_bot.persistence.mongo_repository import MongoRepository
from trading_bot.persistence.redis_repository import RedisRepository
from trading_bot.persistence.connection_manager import ConnectionManager


class PositionModel:
    """Data model for position persistence"""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        broker: str,
        last_updated: Optional[datetime] = None,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        position_id: Optional[str] = None,
        open_date: Optional[datetime] = None,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _id: Optional[str] = None
    ):
        """
        Initialize a position model.
        
        Args:
            symbol: Asset symbol
            quantity: Position quantity (positive for long, negative for short)
            avg_cost: Average cost basis
            broker: Broker identifier
            last_updated: Last update timestamp
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            position_id: Unique position identifier (defaults to broker:symbol)
            open_date: When the position was opened
            strategy: Strategy that opened the position
            metadata: Additional metadata
            _id: MongoDB document ID
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.broker = broker
        self.last_updated = last_updated or datetime.now()
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.position_id = position_id or f"{broker}:{symbol}"
        self.open_date = open_date or datetime.now()
        self.strategy = strategy
        self.metadata = metadata or {}
        self._id = _id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionModel':
        """Create a PositionModel from a dictionary"""
        # Convert ObjectId to string if present
        if '_id' in data and not isinstance(data['_id'], str):
            data['_id'] = str(data['_id'])
            
        # Handle timestamp conversion
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
        if 'open_date' in data and isinstance(data['open_date'], str):
            data['open_date'] = datetime.fromisoformat(data['open_date'].replace('Z', '+00:00'))
            
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        result = {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'broker': self.broker,
            'last_updated': self.last_updated.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'position_id': self.position_id,
            'strategy': self.strategy,
            'metadata': self.metadata
        }
        
        if self.open_date:
            result['open_date'] = self.open_date.isoformat()
        
        if self._id:
            result['_id'] = self._id
            
        return result
    
    def update_from_fill(self, price: float, quantity: float) -> None:
        """
        Update position from a fill.
        
        Args:
            price: Fill price
            quantity: Fill quantity (positive for buy, negative for sell)
        """
        current_quantity = self.quantity
        current_cost = current_quantity * self.avg_cost
        
        # Calculate new position
        new_quantity = current_quantity + quantity
        
        # If we're closing the position
        if (current_quantity > 0 and new_quantity <= 0) or (current_quantity < 0 and new_quantity >= 0):
            # Calculate realized P&L
            closing_value = abs(quantity) * price
            closing_cost = abs(quantity) * self.avg_cost
            
            if current_quantity > 0:  # Long position
                self.realized_pnl += closing_value - closing_cost
            else:  # Short position
                self.realized_pnl += closing_cost - closing_value
            
            # If fully closed
            if new_quantity == 0:
                self.avg_cost = 0.0
            # If flipped from long to short or vice versa
            else:
                remaining_quantity = abs(new_quantity)
                self.avg_cost = price
        # Adding to position
        elif ((current_quantity >= 0 and quantity > 0) or 
              (current_quantity <= 0 and quantity < 0)):
            # Update average cost
            total_cost = current_cost + (quantity * price)
            self.avg_cost = total_cost / new_quantity
        
        # Update position size
        self.quantity = new_quantity
        self.last_updated = datetime.now()


class PositionRepository:
    """Repository for Position persistence"""
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the position repository.
        
        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.mongo_repo = MongoRepository(connection_manager, 'positions')
        self.redis_repo = RedisRepository(connection_manager, 'positions')
        self.logger = logging.getLogger(__name__)
        
        # Set of positions that need to be synced to MongoDB
        self._dirty_positions: Set[str] = set()
    
    def save_position(self, position: Union[PositionModel, Dict[str, Any]]) -> str:
        """
        Save a position to the database.
        
        Args:
            position: Position model or dictionary to save
            
        Returns:
            Position ID
        """
        if not isinstance(position, PositionModel):
            position = PositionModel.from_dict(position)
            
        # Update timestamp
        position.last_updated = datetime.now()
        
        try:
            # Save to Redis first (fast cache)
            try:
                self.redis_repo.save(position)
            except Exception as e:
                self.logger.warning(f"Failed to save position to Redis cache: {str(e)}")
            
            # Save to MongoDB (durable storage)
            self.mongo_repo.save(position)
            
            return position.position_id
            
        except Exception as e:
            self.logger.error(f"Failed to save position: {str(e)}")
            raise
    
    def update_position(self, position_id: str, position: Union[PositionModel, Dict[str, Any]]) -> bool:
        """
        Update a position in the database.
        
        Args:
            position_id: Position ID
            position: Updated position
            
        Returns:
            True if successful
        """
        if not isinstance(position, PositionModel):
            position = PositionModel.from_dict(position)
            
        # Update timestamp
        position.last_updated = datetime.now()
        
        try:
            # Update in Redis first (fast cache)
            try:
                self.redis_repo.update(position_id, position)
            except Exception as e:
                self.logger.warning(f"Failed to update position in Redis cache: {str(e)}")
            
            # Find MongoDB document ID
            mongo_position = self.find_by_position_id(position_id)
            
            if mongo_position and mongo_position._id:
                # Update in MongoDB
                self.mongo_repo.update(mongo_position._id, position)
            else:
                # Save as new
                self.mongo_repo.save(position)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update position: {str(e)}")
            raise
    
    def find_by_position_id(self, position_id: str) -> Optional[PositionModel]:
        """
        Find a position by position ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position if found, None otherwise
        """
        try:
            # Try Redis first
            try:
                position = self.redis_repo.find_by_id(position_id)
                if position:
                    if isinstance(position, dict):
                        return PositionModel.from_dict(position)
                    return position
            except Exception:
                # Redis error, continue to MongoDB
                pass
            
            # Try MongoDB
            positions = self.mongo_repo.find_by_query({'position_id': position_id})
            
            if positions and len(positions) > 0:
                if isinstance(positions[0], dict):
                    return PositionModel.from_dict(positions[0])
                return positions[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find position: {str(e)}")
            raise
    
    def find_by_symbol(self, symbol: str) -> List[PositionModel]:
        """
        Find positions by symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            List of matching positions
        """
        try:
            # Try MongoDB
            positions = self.mongo_repo.find_by_query({'symbol': symbol})
            
            # Convert to PositionModel if needed
            result = []
            for position in positions:
                if isinstance(position, dict):
                    result.append(PositionModel.from_dict(position))
                else:
                    result.append(position)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find positions by symbol: {str(e)}")
            raise
    
    def find_by_broker(self, broker: str) -> List[PositionModel]:
        """
        Find positions by broker.
        
        Args:
            broker: Broker identifier
            
        Returns:
            List of matching positions
        """
        try:
            # Try MongoDB
            positions = self.mongo_repo.find_by_query({'broker': broker})
            
            # Convert to PositionModel if needed
            result = []
            for position in positions:
                if isinstance(position, dict):
                    result.append(PositionModel.from_dict(position))
                else:
                    result.append(position)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find positions by broker: {str(e)}")
            raise
    
    def find_all_positions(self) -> List[PositionModel]:
        """
        Find all positions.
        
        Returns:
            List of all positions
        """
        try:
            # Try Redis first for hot list
            try:
                positions = self.redis_repo.find_all()
                if positions:
                    # Convert to PositionModel if needed
                    result = []
                    for position in positions:
                        if isinstance(position, dict):
                            result.append(PositionModel.from_dict(position))
                        else:
                            result.append(position)
                    return result
            except Exception:
                # Redis error, continue to MongoDB
                pass
            
            # Fall back to MongoDB
            positions = self.mongo_repo.find_all()
            
            # Convert to PositionModel if needed
            result = []
            for position in positions:
                if isinstance(position, dict):
                    result.append(PositionModel.from_dict(position))
                else:
                    result.append(position)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find all positions: {str(e)}")
            raise
    
    def find_non_zero_positions(self) -> List[PositionModel]:
        """
        Find all non-zero positions.
        
        Returns:
            List of positions with non-zero quantity
        """
        try:
            # Try MongoDB
            positions = self.mongo_repo.find_by_query({'quantity': {'$ne': 0}})
            
            # Convert to PositionModel if needed
            result = []
            for position in positions:
                if isinstance(position, dict):
                    result.append(PositionModel.from_dict(position))
                else:
                    result.append(position)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find non-zero positions: {str(e)}")
            raise
    
    def update_from_fill(self, symbol: str, broker: str, price: float, quantity: float, 
                         strategy: Optional[str] = None) -> PositionModel:
        """
        Update position from a fill.
        
        Args:
            symbol: Asset symbol
            broker: Broker identifier
            price: Fill price
            quantity: Fill quantity (positive for buy, negative for sell)
            strategy: Strategy name (optional)
            
        Returns:
            Updated position
        """
        try:
            # Generate position ID
            position_id = f"{broker}:{symbol}"
            
            # Find existing position
            position = self.find_by_position_id(position_id)
            
            if not position:
                # Create new position
                position = PositionModel(
                    symbol=symbol,
                    quantity=0,
                    avg_cost=0,
                    broker=broker,
                    open_date=datetime.now(),
                    strategy=strategy
                )
            
            # Update position
            position.update_from_fill(price, quantity)
            
            # If strategy is provided, update it
            if strategy and not position.strategy:
                position.strategy = strategy
            
            # Save position
            if position._id:
                self.update_position(position_id, position)
            else:
                self.save_position(position)
            
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to update position from fill: {str(e)}")
            raise
    
    def delete_position(self, position_id: str) -> bool:
        """
        Delete a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            True if successful
        """
        try:
            # Delete from Redis
            try:
                self.redis_repo.delete(position_id)
            except Exception as e:
                self.logger.warning(f"Failed to delete position from Redis cache: {str(e)}")
            
            # Find MongoDB document ID
            position = self.find_by_position_id(position_id)
            
            if position and position._id:
                # Delete from MongoDB
                self.mongo_repo.delete(position._id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete position: {str(e)}")
            raise
    
    def load_positions(self, positions: List[Union[PositionModel, Dict[str, Any]]]) -> None:
        """
        Load multiple positions into the repository.
        
        Args:
            positions: List of positions to load
        """
        try:
            # Convert to PositionModel if needed
            position_models = []
            for position in positions:
                if isinstance(position, dict):
                    position_models.append(PositionModel.from_dict(position))
                else:
                    position_models.append(position)
            
            # Save each position to Redis
            for position in position_models:
                try:
                    self.redis_repo.save(position)
                except Exception as e:
                    self.logger.warning(f"Failed to load position to Redis cache: {str(e)}")
            
            # Log summary
            self.logger.info(f"Loaded {len(position_models)} positions into repository")
            
        except Exception as e:
            self.logger.error(f"Failed to load positions: {str(e)}")
            raise
    
    def sync_to_durable_storage(self) -> int:
        """
        Sync positions from Redis to MongoDB for durability.
        
        Returns:
            Number of positions synced
        """
        try:
            # Get all positions from Redis
            redis_positions = self.redis_repo.find_all()
            
            if not redis_positions:
                return 0
            
            # Sync each position to MongoDB
            synced_count = 0
            for position in redis_positions:
                try:
                    if isinstance(position, dict):
                        position = PositionModel.from_dict(position)
                    
                    # Find MongoDB document ID
                    mongo_position = self.find_by_position_id(position.position_id)
                    
                    if mongo_position and mongo_position._id:
                        # Update in MongoDB
                        self.mongo_repo.update(mongo_position._id, position)
                    else:
                        # Save as new
                        self.mongo_repo.save(position)
                    
                    synced_count += 1
                
                except Exception as e:
                    self.logger.error(f"Failed to sync position {position.position_id}: {str(e)}")
            
            self.logger.info(f"Synced {synced_count} positions to durable storage")
            return synced_count
            
        except Exception as e:
            self.logger.error(f"Failed to sync positions to durable storage: {str(e)}")
            raise
