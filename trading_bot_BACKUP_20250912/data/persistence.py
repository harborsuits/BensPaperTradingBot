#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Persistence layer for BensBot using MongoDB
Allows saving and retrieving trades, strategy states, and performance metrics
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

class PersistenceManager:
    """
    MongoDB-based persistence manager for BensBot
    Handles storage and retrieval of trades, strategy states, performance metrics,
    and system logs to enable crash recovery and performance analysis
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                database: str = "bensbot", auto_connect: bool = True):
        """
        Initialize the persistence manager with MongoDB connection
        
        Args:
            connection_string: MongoDB connection string
            database: Database name to use
            auto_connect: Whether to connect immediately
        """
        self.connection_string = connection_string
        self.database_name = database
        self.client = None
        self.db = None
        self.connected = False
        
        # Initialize collections dict
        self.collections = {}
        
        # Connect if auto_connect is True
        if auto_connect:
            self.connect()
            
    def connect(self) -> bool:
        """
        Connect to MongoDB database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Check if connection is successful by running a command
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.connected = True
            logger.info(f"Successfully connected to MongoDB: {self.database_name}")
            
            # Initialize collections with indexes
            self._initialize_collections()
            
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.connected = False
            return False
            
    def disconnect(self) -> None:
        """Disconnect from MongoDB database"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.connected = False
            logger.info("Disconnected from MongoDB")
            
    def _initialize_collections(self) -> None:
        """Initialize collections and create indexes"""
        # Trades collection
        trades_collection = self.db.trades
        trades_collection.create_index([("symbol", ASCENDING)])
        trades_collection.create_index([("strategy_id", ASCENDING)])
        trades_collection.create_index([("created_at", DESCENDING)])
        trades_collection.create_index([("status", ASCENDING)])
        
        # Strategy states collection
        strategy_states_collection = self.db.strategy_states
        strategy_states_collection.create_index([("strategy_id", ASCENDING)], unique=True)
        strategy_states_collection.create_index([("last_updated", DESCENDING)])
        
        # Performance metrics collection
        performance_collection = self.db.performance
        performance_collection.create_index([("timestamp", DESCENDING)])
        performance_collection.create_index([("strategy_id", ASCENDING)])
        
        # System logs collection
        logs_collection = self.db.system_logs
        logs_collection.create_index([("timestamp", DESCENDING)])
        logs_collection.create_index([("level", ASCENDING)])
        logs_collection.create_index([("component", ASCENDING)])
        
        # Store references
        self.collections = {
            'trades': trades_collection,
            'strategy_states': strategy_states_collection,
            'performance': performance_collection,
            'logs': logs_collection
        }
        
        logger.debug("MongoDB collections initialized with indexes")
        
    def is_connected(self) -> bool:
        """Check if connected to MongoDB"""
        if not self.client:
            return False
            
        try:
            # Check connection with a simple command
            self.client.admin.command('ping')
            return True
        except:
            return False
            
    # ---- Trade Management ----
            
    def save_trade(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """
        Persist trade to database with timestamp and status tracking
        
        Args:
            trade_data: Dictionary containing trade data
            
        Returns:
            str: ID of inserted document, or None if unsuccessful
        """
        if not self.is_connected():
            logger.error("Cannot save trade: Not connected to database")
            return None
            
        collection = self.collections.get('trades')
        if collection is None:
            logger.error("Trades collection not initialized")
            return None
            
        # Add timestamps
        now = datetime.now()
        if '_id' not in trade_data:
            trade_data['created_at'] = now
            trade_data['last_updated'] = now
        else:
            trade_data['last_updated'] = now
            
        try:
            if '_id' in trade_data and trade_data['_id']:
                # Update existing trade
                result = collection.replace_one(
                    {'_id': trade_data['_id']}, 
                    trade_data
                )
                return str(trade_data['_id'])
            else:
                # Insert new trade
                result = collection.insert_one(trade_data)
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving trade: {str(e)}")
            return None
            
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific trade by ID
        
        Args:
            trade_id: ID of the trade to retrieve
            
        Returns:
            Dict or None: Trade data dictionary or None if not found
        """
        if not self.is_connected():
            logger.error("Cannot get trade: Not connected to database")
            return None
            
        collection = self.collections.get('trades')
        if collection is None:
            logger.error("Trades collection not initialized")
            return None
            
        try:
            from bson.objectid import ObjectId
            trade = collection.find_one({'_id': ObjectId(trade_id)})
            return trade
        except Exception as e:
            logger.error(f"Error retrieving trade: {str(e)}")
            return None
            
    def get_trades_history(self, symbol: Optional[str] = None, 
                          strategy_id: Optional[str] = None,
                          status: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 100) -> pd.DataFrame:
        """
        Retrieve historical trades with optional filtering
        
        Args:
            symbol: Filter by symbol
            strategy_id: Filter by strategy ID
            status: Filter by trade status
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame: DataFrame containing trade data
        """
        if not self.is_connected():
            logger.error("Cannot get trade history: Not connected to database")
            return pd.DataFrame()
            
        collection = self.collections.get('trades')
        if collection is None:
            logger.error("Trades collection not initialized")
            return pd.DataFrame()
            
        # Build query
        query = {}
        if symbol:
            query['symbol'] = symbol
        if strategy_id:
            query['strategy_id'] = strategy_id
        if status:
            query['status'] = status
            
        # Date range query
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query['$gte'] = start_date
            if end_date:
                date_query['$lte'] = end_date
                
            if date_query:
                query['created_at'] = date_query
                
        try:
            # Execute query
            cursor = collection.find(query).sort('created_at', -1).limit(limit)
            trades = list(cursor)
            
            if not trades:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            return df
        except Exception as e:
            logger.error(f"Error retrieving trade history: {str(e)}")
            return pd.DataFrame()
            
    def update_trade_status(self, trade_id: str, status: str, 
                          additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a trade
        
        Args:
            trade_id: ID of the trade to update
            status: New status value
            additional_data: Additional data to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot update trade: Not connected to database")
            return False
            
        collection = self.collections.get('trades')
        if collection is None:
            logger.error("Trades collection not initialized")
            return False
            
        try:
            from bson.objectid import ObjectId
            update_data = {'$set': {'status': status, 'last_updated': datetime.now()}}
            
            # Add additional data if provided
            if additional_data:
                for key, value in additional_data.items():
                    update_data['$set'][key] = value
                    
            result = collection.update_one(
                {'_id': ObjectId(trade_id)},
                update_data
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating trade status: {str(e)}")
            return False
            
    # ---- Strategy State Management ----
    
    def save_strategy_state(self, strategy_id: str, state: Dict[str, Any]) -> bool:
        """
        Save current state of a strategy for recovery
        
        Args:
            strategy_id: Strategy identifier
            state: Dictionary containing strategy state
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot save strategy state: Not connected to database")
            return False
            
        collection = self.collections.get('strategy_states')
        if collection is None:
            logger.error("Strategy states collection not initialized")
            return False
            
        try:
            # Set metadata
            state['strategy_id'] = strategy_id
            state['last_updated'] = datetime.now()
            
            # Use upsert to create or update
            result = collection.replace_one(
                {'strategy_id': strategy_id}, 
                state, 
                upsert=True
            )
            
            return True
        except Exception as e:
            logger.error(f"Error saving strategy state: {str(e)}")
            return False
            
    def load_strategy_state(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Load saved state for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict or None: Strategy state dictionary or None if not found
        """
        if not self.is_connected():
            logger.error("Cannot load strategy state: Not connected to database")
            return None
            
        collection = self.collections.get('strategy_states')
        if collection is None:
            logger.error("Strategy states collection not initialized")
            return None
            
        try:
            state = collection.find_one({'strategy_id': strategy_id})
            return state
        except Exception as e:
            logger.error(f"Error loading strategy state: {str(e)}")
            return None
            
    def delete_strategy_state(self, strategy_id: str) -> bool:
        """
        Delete a strategy state
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot delete strategy state: Not connected to database")
            return False
            
        collection = self.collections.get('strategy_states')
        if collection is None:
            logger.error("Strategy states collection not initialized")
            return False
            
        try:
            result = collection.delete_one({'strategy_id': strategy_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting strategy state: {str(e)}")
            return False
            
    # ---- Performance Metrics ----
    
    def save_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Store performance metrics with timestamps for tracking
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot save performance metrics: Not connected to database")
            return False
            
        collection = self.collections.get('performance')
        if collection is None:
            logger.error("Performance collection not initialized")
            return False
            
        try:
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now()
                
            result = collection.insert_one(metrics)
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
            return False
            
    def get_performance_history(self, strategy_id: Optional[str] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              metrics: Optional[List[str]] = None,
                              limit: int = 100) -> pd.DataFrame:
        """
        Retrieve performance metrics history with optional filtering
        
        Args:
            strategy_id: Filter by strategy ID
            start_date: Filter by start date
            end_date: Filter by end date
            metrics: List of specific metrics to retrieve
            limit: Maximum number of records to return
            
        Returns:
            DataFrame: DataFrame containing performance metrics
        """
        if not self.is_connected():
            logger.error("Cannot get performance history: Not connected to database")
            return pd.DataFrame()
            
        collection = self.collections.get('performance')
        if collection is None:
            logger.error("Performance collection not initialized")
            return pd.DataFrame()
            
        # Build query
        query = {}
        if strategy_id:
            query['strategy_id'] = strategy_id
            
        # Date range query
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query['$gte'] = start_date
            if end_date:
                date_query['$lte'] = end_date
                
            if date_query:
                query['timestamp'] = date_query
                
        try:
            # Define projection if specific metrics requested
            projection = None
            if metrics:
                projection = {metric: 1 for metric in metrics}
                # Always include timestamp and strategy_id
                projection['timestamp'] = 1
                projection['strategy_id'] = 1
                
            # Execute query
            cursor = collection.find(
                query, 
                projection
            ).sort('timestamp', -1).limit(limit)
            
            metrics_data = list(cursor)
            
            if not metrics_data:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(metrics_data)
            return df
        except Exception as e:
            logger.error(f"Error retrieving performance history: {str(e)}")
            return pd.DataFrame()
            
    # ---- System Logs ----
    
    def log_system_event(self, level: str, message: str, 
                        component: str, additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a system event for auditing and debugging
        
        Args:
            level: Log level (e.g., 'INFO', 'WARNING', 'ERROR')
            message: Log message
            component: System component that generated the log
            additional_data: Additional contextual data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Cannot log system event: Not connected to database")
            return False
            
        collection = self.collections.get('logs')
        if collection is None:
            logger.error("Logs collection not initialized")
            return False
            
        try:
            log_entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'component': component
            }
            
            # Add additional data if provided
            if additional_data:
                log_entry['data'] = additional_data
                
            result = collection.insert_one(log_entry)
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error logging system event: {str(e)}")
            return False
            
    def get_system_logs(self, level: Optional[str] = None,
                      component: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 100) -> pd.DataFrame:
        """
        Retrieve system logs with optional filtering
        
        Args:
            level: Filter by log level
            component: Filter by component
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of logs to return
            
        Returns:
            DataFrame: DataFrame containing log entries
        """
        if not self.is_connected():
            logger.error("Cannot get system logs: Not connected to database")
            return pd.DataFrame()
            
        collection = self.collections.get('logs')
        if collection is None:
            logger.error("Logs collection not initialized")
            return pd.DataFrame()
            
        # Build query
        query = {}
        if level:
            query['level'] = level
        if component:
            query['component'] = component
            
        # Date range query
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query['$gte'] = start_date
            if end_date:
                date_query['$lte'] = end_date
                
            if date_query:
                query['timestamp'] = date_query
                
        try:
            # Execute query
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            logs = list(cursor)
            
            if not logs:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(logs)
            return df
        except Exception as e:
            logger.error(f"Error retrieving system logs: {str(e)}")
            return pd.DataFrame()
            
    # ---- Database Management ----
    
    def create_backup(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the database
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            bool: True if successful, False otherwise
        """
        # This would typically use mongodump, but for simplicity
        # we'll just log a message
        logger.info(f"Database backup requested to path: {backup_path}")
        
        # This would call a subprocess to run mongodump
        # For now, just a placeholder
        return True
        
    def delete_old_data(self, collection_name: str, 
                       older_than_days: int, 
                       dry_run: bool = True) -> int:
        """
        Delete data older than specified days
        
        Args:
            collection_name: Name of collection to clean
            older_than_days: Delete data older than this many days
            dry_run: If True, only count records that would be deleted
            
        Returns:
            int: Number of records deleted (or would be deleted in dry run)
        """
        if not self.is_connected():
            logger.error("Cannot delete old data: Not connected to database")
            return 0
            
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not initialized")
            return 0
            
        collection = self.collections[collection_name]
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        # Find the timestamp field for this collection
        timestamp_field = 'timestamp'
        if collection_name == 'trades':
            timestamp_field = 'created_at'
        elif collection_name == 'strategy_states':
            timestamp_field = 'last_updated'
            
        # Build query
        query = {timestamp_field: {'$lt': cutoff_date}}
        
        try:
            # Count records that would be deleted
            count = collection.count_documents(query)
            
            if not dry_run:
                # Actually delete the records
                result = collection.delete_many(query)
                logger.info(f"Deleted {result.deleted_count} records from {collection_name}")
                return result.deleted_count
            else:
                logger.info(f"Dry run: Would delete {count} records from {collection_name}")
                return count
        except Exception as e:
            logger.error(f"Error deleting old data: {str(e)}")
            return 0
