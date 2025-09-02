#!/usr/bin/env python3
"""
Idempotency Module

This module provides an idempotency decorator for broker operations
to ensure operations like order placement and cancellation are not duplicated.
"""

import logging
import functools
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, List, TypeVar, cast

from trading_bot.persistence.mongo_repository import MongoRepository
from trading_bot.persistence.connection_manager import ConnectionManager

# Type variables for decorator
F = TypeVar('F', bound=Callable[..., Any])

class IdempotencyManager:
    """Manages idempotent operations using MongoDB for persistence"""
    
    def __init__(self, connection_manager: ConnectionManager, ttl_days: int = 30):
        """
        Initialize the idempotency manager.
        
        Args:
            connection_manager: Database connection manager
            ttl_days: Time-to-live in days for idempotency records
        """
        self.connection_manager = connection_manager
        self.mongo_repo = MongoRepository(connection_manager, 'idempotency')
        self.logger = logging.getLogger(__name__)
        self.ttl_days = ttl_days
        
        # In-memory cache of idempotency keys for faster lookups
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize by loading active keys to cache
        self._init_cache()
    
    def _init_cache(self) -> None:
        """Initialize the cache by loading active idempotency records"""
        try:
            # Get active records (not expired)
            now = datetime.now()
            min_date = now - timedelta(days=self.ttl_days)
            
            records = self.mongo_repo.find_by_query({
                'created_at': {'$gte': min_date.isoformat()}
            })
            
            for record in records:
                if isinstance(record, dict):
                    key = record.get('idempotency_key')
                    if key:
                        self.cache[key] = record
                
            self.logger.info(f"Loaded {len(self.cache)} idempotency records into cache")
            
        except Exception as e:
            self.logger.error(f"Error initializing idempotency cache: {str(e)}")
    
    def register_operation(self, operation_type: str, broker: str, params: Dict[str, Any]) -> str:
        """
        Register an operation and get an idempotency key.
        
        Args:
            operation_type: Type of operation (e.g., 'place_order', 'cancel_order')
            broker: Broker identifier
            params: Operation parameters
            
        Returns:
            Idempotency key (UUID)
        """
        # Generate idempotency key
        idempotency_key = str(uuid.uuid4())
        
        # Create record
        record = {
            'idempotency_key': idempotency_key,
            'operation_type': operation_type,
            'broker': broker,
            'params': params,
            'created_at': datetime.now().isoformat(),
            'result': None
        }
        
        try:
            # Save to MongoDB
            self.mongo_repo.save(record)
            
            # Update cache
            self.cache[idempotency_key] = record
            
            return idempotency_key
            
        except Exception as e:
            self.logger.error(f"Error registering idempotent operation: {str(e)}")
            raise
    
    def get_operation(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """
        Get an operation by idempotency key.
        
        Args:
            idempotency_key: Idempotency key
            
        Returns:
            Operation record if found
        """
        # Check cache first
        if idempotency_key in self.cache:
            return self.cache[idempotency_key]
            
        try:
            # Try MongoDB
            operations = self.mongo_repo.find_by_query({'idempotency_key': idempotency_key})
            
            if operations and len(operations) > 0:
                # Update cache
                self.cache[idempotency_key] = operations[0]
                return operations[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting idempotent operation: {str(e)}")
            return None
    
    def record_result(self, idempotency_key: str, result: Any) -> bool:
        """
        Record the result of an idempotent operation.
        
        Args:
            idempotency_key: Idempotency key
            result: Operation result
            
        Returns:
            True if successful
        """
        try:
            # Get operation
            operation = self.get_operation(idempotency_key)
            
            if not operation:
                self.logger.warning(f"Cannot record result for unknown operation: {idempotency_key}")
                return False
                
            # Update operation
            operation['result'] = result
            operation['completed_at'] = datetime.now().isoformat()
            
            # Find MongoDB document ID
            if isinstance(operation, dict) and '_id' in operation:
                mongo_id = operation['_id']
                
                # Update in MongoDB
                self.mongo_repo.update(mongo_id, operation)
                
                # Update cache
                self.cache[idempotency_key] = operation
                
                return True
            else:
                self.logger.warning(f"Cannot update operation without MongoDB ID: {idempotency_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error recording idempotent operation result: {str(e)}")
            return False
    
    def find_by_operation_type(self, operation_type: str, broker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find operations by type.
        
        Args:
            operation_type: Type of operation
            broker: Optional broker filter
            
        Returns:
            List of matching operations
        """
        try:
            # Build query
            query = {'operation_type': operation_type}
            if broker:
                query['broker'] = broker
                
            # Query MongoDB
            operations = self.mongo_repo.find_by_query(query)
            
            return operations
            
        except Exception as e:
            self.logger.error(f"Error finding idempotent operations by type: {str(e)}")
            return []
    
    def find_pending_operations(self, operation_type: Optional[str] = None, 
                              broker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find pending operations (no result recorded).
        
        Args:
            operation_type: Optional operation type filter
            broker: Optional broker filter
            
        Returns:
            List of pending operations
        """
        try:
            # Build query
            query = {'result': None}
            
            if operation_type:
                query['operation_type'] = operation_type
                
            if broker:
                query['broker'] = broker
                
            # Query MongoDB
            operations = self.mongo_repo.find_by_query(query)
            
            return operations
            
        except Exception as e:
            self.logger.error(f"Error finding pending idempotent operations: {str(e)}")
            return []
    
    def cleanup_expired_records(self) -> int:
        """
        Clean up expired idempotency records.
        
        Returns:
            Number of records deleted
        """
        try:
            # Calculate expiration date
            now = datetime.now()
            expiration_date = now - timedelta(days=self.ttl_days)
            
            # Find expired records
            expired_records = self.mongo_repo.find_by_query({
                'created_at': {'$lt': expiration_date.isoformat()}
            })
            
            # Delete expired records
            deleted_count = 0
            for record in expired_records:
                if isinstance(record, dict) and '_id' in record:
                    self.mongo_repo.delete(record['_id'])
                    
                    # Remove from cache if present
                    key = record.get('idempotency_key')
                    if key and key in self.cache:
                        del self.cache[key]
                        
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} expired idempotency records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired idempotency records: {str(e)}")
            return 0


def idempotent(operation_type: str, param_keys: List[str] = None) -> Callable[[F], F]:
    """
    Decorator to make a broker operation idempotent.
    
    Args:
        operation_type: Type of operation (e.g., 'place_order', 'cancel_order')
        param_keys: List of parameter keys to use for idempotency checking
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get first arg as self (instance)
            instance = args[0]
            
            # Check if instance has idempotency_manager attribute
            if not hasattr(instance, 'idempotency_manager'):
                # If no idempotency manager available, just call the function
                logging.warning(f"No idempotency manager available for {operation_type}, proceeding without idempotency")
                return func(*args, **kwargs)
            
            idempotency_manager = instance.idempotency_manager
            
            # Get broker identifier (assuming it's a property or attribute of the instance)
            broker = instance.broker_id if hasattr(instance, 'broker_id') else None
            
            # If idempotency key is provided in kwargs, use it
            idempotency_key = kwargs.get('idempotency_key')
            
            # If no idempotency key provided, check if a previous matching operation exists
            if not idempotency_key and param_keys:
                # Extract relevant parameters
                params = {}
                for key in param_keys:
                    if key in kwargs:
                        params[key] = kwargs[key]
                
                # TODO: Implement matching logic using parameters
                # This would require more complex matching on stored operations
                pass
            
            if idempotency_key:
                # Check if operation already exists and has a result
                operation = idempotency_manager.get_operation(idempotency_key)
                
                if operation and operation.get('result') is not None:
                    # Operation already completed, return stored result
                    logging.info(f"Using stored result for idempotent operation: {idempotency_key}")
                    return operation.get('result')
            else:
                # Generate idempotency key for new operation
                params = {k: v for k, v in kwargs.items() if not k.startswith('_')}
                idempotency_key = idempotency_manager.register_operation(
                    operation_type, broker, params
                )
                # Add idempotency_key to kwargs
                kwargs['idempotency_key'] = idempotency_key
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Record the result
            idempotency_manager.record_result(idempotency_key, result)
            
            return result
        
        return cast(F, wrapper)
    
    return decorator
