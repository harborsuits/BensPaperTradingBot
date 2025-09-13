#!/usr/bin/env python3
"""
Database Connection Manager

This module provides connection management for MongoDB and Redis databases.
"""

import logging
import os
from typing import Dict, Any, Optional
import pymongo
import redis
from pymongo import MongoClient
from pymongo.database import Database
from redis import Redis


class ConnectionManager:
    """
    Manages database connections for MongoDB and Redis.
    
    This class handles connection pooling, configuration, and lifecycle management
    for both MongoDB (durable storage) and Redis (hot state).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the connection manager.
        
        Args:
            config: Configuration dictionary for database connections
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection instances
        self._mongo_client = None
        self._mongo_db = None
        self._redis_client = None
        
        # Connection status tracking
        self.mongo_connected = False
        self.redis_connected = False
    
    def get_mongo_client(self) -> MongoClient:
        """
        Get a MongoDB client instance.
        
        Returns:
            MongoDB client instance
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._mongo_client is None:
            self._connect_mongo()
        
        return self._mongo_client
    
    def get_mongo_db(self) -> Database:
        """
        Get a MongoDB database instance.
        
        Returns:
            MongoDB database instance
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._mongo_db is None:
            self._connect_mongo()
        
        return self._mongo_db
    
    def get_redis_client(self) -> Redis:
        """
        Get a Redis client instance.
        
        Returns:
            Redis client instance
            
        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._redis_client is None:
            self._connect_redis()
        
        return self._redis_client
    
    def _connect_mongo(self) -> None:
        """
        Establish MongoDB connection.
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        mongo_config = self.config.get('mongodb', {})
        
        # Get connection settings with defaults
        uri = mongo_config.get('uri', 'mongodb://localhost:27017')
        db_name = mongo_config.get('db_name', 'bensbot')
        
        try:
            self.logger.info(f"Connecting to MongoDB at {uri}")
            self._mongo_client = MongoClient(uri)
            self._mongo_db = self._mongo_client[db_name]
            
            # Test the connection
            self._mongo_client.admin.command('ping')
            self.mongo_connected = True
            self.logger.info(f"Successfully connected to MongoDB database: {db_name}")
            
        except Exception as e:
            self.mongo_connected = False
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def _connect_redis(self) -> None:
        """
        Establish Redis connection.
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        redis_config = self.config.get('redis', {})
        
        # Get connection settings with defaults
        host = redis_config.get('host', 'localhost')
        port = redis_config.get('port', 6379)
        db = redis_config.get('db', 0)
        password = redis_config.get('password', None)
        
        try:
            self.logger.info(f"Connecting to Redis at {host}:{port}")
            self._redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            
            # Test the connection
            self._redis_client.ping()
            self.redis_connected = True
            self.logger.info(f"Successfully connected to Redis, db: {db}")
            
        except Exception as e:
            self.redis_connected = False
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    def close(self) -> None:
        """Close all database connections."""
        if self._mongo_client:
            self.logger.info("Closing MongoDB connection")
            self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
            self.mongo_connected = False
        
        if self._redis_client:
            self.logger.info("Closing Redis connection")
            self._redis_client.close()
            self._redis_client = None
            self.redis_connected = False
