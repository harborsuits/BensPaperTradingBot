#!/usr/bin/env python3
"""
Redis Repository Implementation

This module provides a Redis repository implementation for hot-state caching.
"""

import logging
import json
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type, Union
import redis

from trading_bot.persistence.base_repository import BaseRepository
from trading_bot.persistence.connection_manager import ConnectionManager

# Define generic type for models
T = TypeVar('T')


class RedisRepository(BaseRepository[T]):
    """
    Redis implementation of the BaseRepository.
    
    This class provides Redis-specific implementation for fast hot-state caching,
    with automatic serialization and deserialization of entities.
    """
    
    def __init__(self, connection_manager: ConnectionManager, key_prefix: str):
        """
        Initialize the Redis repository.
        
        Args:
            connection_manager: Database connection manager
            key_prefix: Prefix for all Redis keys
        """
        super().__init__(connection_manager)
        self.key_prefix = key_prefix
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def redis(self) -> redis.Redis:
        """Get the Redis client"""
        return self.connection_manager.get_redis_client()
    
    def _make_key(self, id: str) -> str:
        """
        Create a full Redis key using the prefix.
        
        Args:
            id: Identifier for the entity
            
        Returns:
            Full Redis key
        """
        return f"{self.key_prefix}:{id}"
    
    def to_json(self, entity: T) -> str:
        """
        Convert an entity to JSON for Redis storage.
        
        Args:
            entity: Entity to convert
            
        Returns:
            JSON string representation of the entity
        """
        if isinstance(entity, dict):
            return json.dumps(entity)
        elif hasattr(entity, 'to_dict') and callable(getattr(entity, 'to_dict')):
            return json.dumps(entity.to_dict())
        else:
            raise ValueError(f"Cannot convert entity of type {type(entity).__name__} to JSON")
    
    def from_json(self, json_str: str) -> T:
        """
        Convert a JSON string from Redis to an entity.
        
        Args:
            json_str: JSON string to convert
            
        Returns:
            Entity created from the JSON string
        """
        # Default implementation just parses JSON to dict
        # Subclasses should override this to create proper entity objects
        return json.loads(json_str)  # type: ignore
    
    def save(self, entity: T) -> str:
        """
        Save an entity to Redis.
        
        Args:
            entity: Entity to save
            
        Returns:
            ID of the saved entity
        """
        try:
            # Get ID from entity if available
            entity_id = ""
            if isinstance(entity, dict) and 'id' in entity:
                entity_id = entity['id']
            elif isinstance(entity, dict) and '_id' in entity:
                entity_id = entity['_id']
            elif hasattr(entity, 'id') and entity.id:
                entity_id = entity.id
            
            if not entity_id:
                raise ValueError("Entity must have an id to be saved in Redis")
            
            # Convert entity to JSON and save
            json_str = self.to_json(entity)
            key = self._make_key(entity_id)
            self.redis.set(key, json_str)
            
            self.logger.debug(f"Saved entity to Redis with key {key}")
            return entity_id
            
        except Exception as e:
            self.logger.error(f"Failed to save entity to Redis: {str(e)}")
            raise
    
    def update(self, id: str, entity: T) -> bool:
        """
        Update an entity in Redis (same as save for Redis).
        
        Args:
            id: ID of the entity to update
            entity: Updated entity
            
        Returns:
            True if the update was successful
        """
        try:
            # Convert entity to JSON and save
            json_str = self.to_json(entity)
            key = self._make_key(id)
            self.redis.set(key, json_str)
            
            self.logger.debug(f"Updated entity in Redis with key {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update entity in Redis: {str(e)}")
            raise
    
    def find_by_id(self, id: str) -> Optional[T]:
        """
        Find an entity by ID in Redis.
        
        Args:
            id: ID of the entity to find
            
        Returns:
            The entity if found, None otherwise
        """
        try:
            key = self._make_key(id)
            json_str = self.redis.get(key)
            
            if json_str:
                self.logger.debug(f"Found entity in Redis with key {key}")
                return self.from_json(json_str)
            else:
                self.logger.debug(f"No entity found in Redis with key {key}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to find entity in Redis: {str(e)}")
            raise
    
    def find_all(self) -> List[T]:
        """
        Find all entities in Redis with the key prefix.
        
        Returns:
            List of all entities
        """
        try:
            # Get all keys with the prefix
            keys = self.redis.keys(f"{self.key_prefix}:*")
            
            if not keys:
                return []
            
            # Get all values in a single operation
            values = self.redis.mget(keys)
            
            # Convert values to entities
            entities = []
            for json_str in values:
                if json_str:
                    entities.append(self.from_json(json_str))
            
            self.logger.debug(f"Found {len(entities)} entities in Redis with prefix {self.key_prefix}")
            return entities
            
        except Exception as e:
            self.logger.error(f"Failed to find entities in Redis: {str(e)}")
            raise
    
    def delete(self, id: str) -> bool:
        """
        Delete an entity from Redis.
        
        Args:
            id: ID of the entity to delete
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        try:
            key = self._make_key(id)
            result = self.redis.delete(key)
            
            success = result > 0
            if success:
                self.logger.debug(f"Deleted entity from Redis with key {key}")
            else:
                self.logger.debug(f"No entity found to delete in Redis with key {key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete entity from Redis: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Count the number of entities in Redis with the key prefix.
        
        Returns:
            Number of entities
        """
        try:
            # Get all keys with the prefix
            keys = self.redis.keys(f"{self.key_prefix}:*")
            count = len(keys)
            
            self.logger.debug(f"Counted {count} entities in Redis with prefix {self.key_prefix}")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to count entities in Redis: {str(e)}")
            raise
