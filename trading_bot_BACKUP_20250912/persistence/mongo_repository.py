#!/usr/bin/env python3
"""
MongoDB Repository Implementation

This module provides a base MongoDB repository implementation for all entities.
"""

import logging
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from trading_bot.persistence.base_repository import BaseRepository
from trading_bot.persistence.connection_manager import ConnectionManager

# Define generic type for models
T = TypeVar('T')


class MongoRepository(BaseRepository[T]):
    """
    MongoDB implementation of the BaseRepository.
    
    This class provides the MongoDB-specific implementation of the repository
    interface, handling all database operations against MongoDB collections.
    """
    
    def __init__(self, connection_manager: ConnectionManager, collection_name: str):
        """
        Initialize the MongoDB repository.
        
        Args:
            connection_manager: MongoDB connection manager
            collection_name: Name of the MongoDB collection
        """
        super().__init__(connection_manager)
        self.collection_name = collection_name
        self._collection = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def collection(self) -> Collection:
        """
        Get the MongoDB collection.
        
        Returns:
            MongoDB collection
        """
        if self._collection is None:
            self._collection = self.connection_manager.get_mongo_db()[self.collection_name]
        return self._collection
    
    def to_dict(self, entity: T) -> Dict[str, Any]:
        """
        Convert an entity to a dictionary for MongoDB storage.
        
        Args:
            entity: Entity to convert
            
        Returns:
            Dictionary representation of the entity
        """
        # Default implementation assumes entity is either a dict or has a to_dict method
        if isinstance(entity, dict):
            return entity
        elif hasattr(entity, 'to_dict') and callable(getattr(entity, 'to_dict')):
            return entity.to_dict()
        else:
            raise ValueError(f"Cannot convert entity of type {type(entity).__name__} to dict")
    
    def from_dict(self, data: Dict[str, Any]) -> T:
        """
        Convert a dictionary from MongoDB to an entity.
        
        Args:
            data: Dictionary to convert
            
        Returns:
            Entity created from the dictionary
        """
        # This is a placeholder that should be overridden by subclasses
        return data  # type: ignore
    
    def save(self, entity: T) -> str:
        """
        Save an entity to MongoDB.
        
        Args:
            entity: Entity to save
            
        Returns:
            ID of the saved entity
            
        Raises:
            PyMongoError: If the save operation fails
        """
        try:
            data = self.to_dict(entity)
            
            # Handle the case where _id is already present
            if '_id' in data and not data['_id']:
                del data['_id']
                
            result = self.collection.insert_one(data)
            self.logger.debug(f"Saved entity to {self.collection_name} with id {result.inserted_id}")
            return str(result.inserted_id)
            
        except PyMongoError as e:
            self.logger.error(f"Failed to save entity to {self.collection_name}: {str(e)}")
            raise
    
    def update(self, id: str, entity: T) -> bool:
        """
        Update an entity in MongoDB.
        
        Args:
            id: ID of the entity to update
            entity: Updated entity
            
        Returns:
            True if the update was successful, False otherwise
            
        Raises:
            PyMongoError: If the update operation fails
        """
        try:
            data = self.to_dict(entity)
            
            # Remove _id if present to avoid conflicts
            if '_id' in data:
                del data['_id']
            
            result = self.collection.update_one(
                {'_id': ObjectId(id)},
                {'$set': data}
            )
            
            self.logger.debug(f"Updated entity in {self.collection_name} with id {id}")
            return result.modified_count > 0
            
        except PyMongoError as e:
            self.logger.error(f"Failed to update entity in {self.collection_name}: {str(e)}")
            raise
    
    def find_by_id(self, id: str) -> Optional[T]:
        """
        Find an entity by ID in MongoDB.
        
        Args:
            id: ID of the entity to find
            
        Returns:
            The entity if found, None otherwise
            
        Raises:
            PyMongoError: If the find operation fails
        """
        try:
            data = self.collection.find_one({'_id': ObjectId(id)})
            
            if data:
                self.logger.debug(f"Found entity in {self.collection_name} with id {id}")
                return self.from_dict(data)
            else:
                self.logger.debug(f"No entity found in {self.collection_name} with id {id}")
                return None
                
        except PyMongoError as e:
            self.logger.error(f"Failed to find entity in {self.collection_name}: {str(e)}")
            raise
    
    def find_all(self) -> List[T]:
        """
        Find all entities in MongoDB.
        
        Returns:
            List of all entities
            
        Raises:
            PyMongoError: If the find operation fails
        """
        try:
            data = list(self.collection.find())
            self.logger.debug(f"Found {len(data)} entities in {self.collection_name}")
            return [self.from_dict(item) for item in data]
            
        except PyMongoError as e:
            self.logger.error(f"Failed to find entities in {self.collection_name}: {str(e)}")
            raise
    
    def find_by_query(self, query: Dict[str, Any]) -> List[T]:
        """
        Find entities by query in MongoDB.
        
        Args:
            query: MongoDB query
            
        Returns:
            List of matching entities
            
        Raises:
            PyMongoError: If the find operation fails
        """
        try:
            data = list(self.collection.find(query))
            self.logger.debug(f"Found {len(data)} entities in {self.collection_name} matching query")
            return [self.from_dict(item) for item in data]
            
        except PyMongoError as e:
            self.logger.error(f"Failed to find entities in {self.collection_name} by query: {str(e)}")
            raise
    
    def delete(self, id: str) -> bool:
        """
        Delete an entity from MongoDB.
        
        Args:
            id: ID of the entity to delete
            
        Returns:
            True if the delete was successful, False otherwise
            
        Raises:
            PyMongoError: If the delete operation fails
        """
        try:
            result = self.collection.delete_one({'_id': ObjectId(id)})
            self.logger.debug(f"Deleted entity from {self.collection_name} with id {id}")
            return result.deleted_count > 0
            
        except PyMongoError as e:
            self.logger.error(f"Failed to delete entity from {self.collection_name}: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Count the number of entities in MongoDB.
        
        Returns:
            Number of entities
            
        Raises:
            PyMongoError: If the count operation fails
        """
        try:
            count = self.collection.count_documents({})
            self.logger.debug(f"Counted {count} entities in {self.collection_name}")
            return count
            
        except PyMongoError as e:
            self.logger.error(f"Failed to count entities in {self.collection_name}: {str(e)}")
            raise
