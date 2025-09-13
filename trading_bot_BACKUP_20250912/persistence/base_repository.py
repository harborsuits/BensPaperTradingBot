#!/usr/bin/env python3
"""
Base Repository Abstract Class

This module defines the base repository interface for all persistence operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic, Type

# Define generic type for models
T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """
    Abstract base class for all repositories.
    
    This class defines the common interface that all repositories must implement,
    providing a consistent API for database operations across the system.
    """
    
    def __init__(self, connection_manager=None):
        """
        Initialize the repository with a connection manager.
        
        Args:
            connection_manager: Database connection manager instance
        """
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def save(self, entity: T) -> str:
        """
        Save an entity to the database.
        
        Args:
            entity: The entity to save
            
        Returns:
            The ID of the saved entity
        """
        pass
    
    @abstractmethod
    def update(self, id: str, entity: T) -> bool:
        """
        Update an existing entity in the database.
        
        Args:
            id: The ID of the entity to update
            entity: The updated entity data
            
        Returns:
            True if the update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def find_by_id(self, id: str) -> Optional[T]:
        """
        Find an entity by its ID.
        
        Args:
            id: The ID of the entity to find
            
        Returns:
            The entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        """
        Retrieve all entities.
        
        Returns:
            A list of all entities
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        Delete an entity from the database.
        
        Args:
            id: The ID of the entity to delete
            
        Returns:
            True if the delete was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Count the number of entities in the database.
        
        Returns:
            The number of entities
        """
        pass
