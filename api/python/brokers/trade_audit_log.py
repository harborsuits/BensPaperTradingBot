"""
Trade Audit Log

Provides persistent logging and audit trail capabilities for all trading operations
and executions across multiple brokers.
"""

import logging
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of events to audit"""
    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"
    ORDER_REPLACED = "order_replaced"
    
    # Account events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    ACCOUNT_UPDATED = "account_updated"
    MARGIN_CALL = "margin_call"
    
    # Broker events
    BROKER_CONNECTED = "broker_connected"
    BROKER_DISCONNECTED = "broker_disconnected"
    BROKER_ERROR = "broker_error"
    BROKER_OPERATION = "broker_operation"
    
    # System events
    STRATEGY_SIGNAL = "strategy_signal"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGE = "config_change"


class TradeAuditLog:
    """
    Abstract base class for trade audit logging
    
    Provides a common interface for different audit log implementations.
    """
    
    def __init__(self):
        """Initialize the audit log"""
        self._lock = threading.RLock()
    
    def log_event(self, event_type: AuditEventType, details: Dict[str, Any],
                broker_id: Optional[str] = None, order_id: Optional[str] = None,
                strategy_id: Optional[str] = None) -> str:
        """
        Log an event to the audit trail
        
        Args:
            event_type: Type of event
            details: Event details
            broker_id: ID of the broker (if applicable)
            order_id: ID of the order (if applicable)
            strategy_id: ID of the strategy (if applicable)
            
        Returns:
            str: Unique event ID
        """
        raise NotImplementedError("Subclasses must implement log_event")
    
    def get_events(self, filters: Optional[Dict[str, Any]] = None,
                limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve events from the audit log
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List[Dict]: List of matching events
        """
        raise NotImplementedError("Subclasses must implement get_events")
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific event by ID
        
        Args:
            event_id: ID of the event
            
        Returns:
            Optional[Dict]: Event details or None if not found
        """
        raise NotImplementedError("Subclasses must implement get_event")
    
    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve the complete history for an order
        
        Args:
            order_id: ID of the order
            
        Returns:
            List[Dict]: All events related to the order
        """
        raise NotImplementedError("Subclasses must implement get_order_history")


class JsonFileAuditLog(TradeAuditLog):
    """
    Stores audit events in JSON files
    
    Each day gets its own JSON file for easier archiving and management.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the JSON file audit log
        
        Args:
            log_dir: Directory to store log files
        """
        super().__init__()
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Index of events for faster lookups
        self._event_index = {}  # event_id -> file_path
        self._order_index = {}  # order_id -> [event_id, ...]
        
        logger.info(f"Initialized JsonFileAuditLog in directory: {log_dir}")
        
        # Load existing indexes from recent files
        self._load_indexes()
    
    def _get_log_file_path(self) -> str:
        """Get the log file path for the current day"""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_log_{today}.json")
    
    def _load_indexes(self, days_back: int = 7) -> None:
        """
        Load event and order indexes from recent log files
        
        Args:
            days_back: Number of days to look back
        """
        with self._lock:
            # Get recent log files
            today = datetime.now().date()
            for i in range(days_back):
                date_str = (today - i * datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                file_path = os.path.join(self.log_dir, f"audit_log_{date_str}.json")
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            events = json.load(f)
                            
                            for event in events:
                                event_id = event.get('event_id')
                                order_id = event.get('order_id')
                                
                                if event_id:
                                    self._event_index[event_id] = file_path
                                
                                if order_id:
                                    if order_id not in self._order_index:
                                        self._order_index[order_id] = []
                                    
                                    if event_id:
                                        self._order_index[order_id].append(event_id)
                    except Exception as e:
                        logger.error(f"Error loading index from file {file_path}: {str(e)}")
    
    def _load_events_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load events from a log file
        
        Args:
            file_path: Path to the log file
            
        Returns:
            List[Dict]: Events from the file
        """
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading events from file {file_path}: {str(e)}")
            return []
    
    def _save_events_to_file(self, file_path: str, events: List[Dict[str, Any]]) -> bool:
        """
        Save events to a log file
        
        Args:
            file_path: Path to the log file
            events: Events to save
            
        Returns:
            bool: Success status
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(events, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving events to file {file_path}: {str(e)}")
            return False
    
    def log_event(self, event_type: AuditEventType, details: Dict[str, Any],
                broker_id: Optional[str] = None, order_id: Optional[str] = None,
                strategy_id: Optional[str] = None) -> str:
        """Log an event to the audit trail"""
        with self._lock:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Create event
            event = {
                'event_id': event_id,
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type.value,
                'details': details
            }
            
            if broker_id:
                event['broker_id'] = broker_id
            
            if order_id:
                event['order_id'] = order_id
            
            if strategy_id:
                event['strategy_id'] = strategy_id
            
            # Get log file path
            file_path = self._get_log_file_path()
            
            # Load existing events
            events = self._load_events_from_file(file_path)
            
            # Add new event
            events.append(event)
            
            # Save events
            if self._save_events_to_file(file_path, events):
                # Update indexes
                self._event_index[event_id] = file_path
                
                if order_id:
                    if order_id not in self._order_index:
                        self._order_index[order_id] = []
                    
                    self._order_index[order_id].append(event_id)
                
                return event_id
            else:
                raise IOError(f"Failed to log event to file: {file_path}")
    
    def get_events(self, filters: Optional[Dict[str, Any]] = None,
                limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve events from the audit log"""
        # Default filters
        if filters is None:
            filters = {}
        
        # Get list of log files to search
        log_files = []
        for file in os.listdir(self.log_dir):
            if file.startswith("audit_log_") and file.endswith(".json"):
                log_files.append(os.path.join(self.log_dir, file))
        
        # Sort files by date (newest first)
        log_files.sort(reverse=True)
        
        # Collect matching events
        all_events = []
        
        for file_path in log_files:
            events = self._load_events_from_file(file_path)
            
            for event in events:
                # Apply filters
                matches = True
                for key, value in filters.items():
                    if key in event:
                        if event[key] != value:
                            matches = False
                            break
                    elif key == 'start_date':
                        if event['timestamp'] < value:
                            matches = False
                            break
                    elif key == 'end_date':
                        if event['timestamp'] > value:
                            matches = False
                            break
                    else:
                        matches = False
                        break
                
                if matches:
                    all_events.append(event)
        
        # Sort by timestamp (newest first)
        all_events.sort(key=lambda e: e['timestamp'], reverse=True)
        
        # Apply pagination
        paginated_events = all_events[offset:offset + limit]
        
        return paginated_events
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific event by ID"""
        with self._lock:
            if event_id not in self._event_index:
                return None
            
            file_path = self._event_index[event_id]
            events = self._load_events_from_file(file_path)
            
            for event in events:
                if event.get('event_id') == event_id:
                    return event
            
            return None
    
    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """Retrieve the complete history for an order"""
        with self._lock:
            if order_id not in self._order_index:
                return []
            
            event_ids = self._order_index[order_id]
            order_events = []
            
            for event_id in event_ids:
                event = self.get_event(event_id)
                if event:
                    order_events.append(event)
            
            # Sort by timestamp
            order_events.sort(key=lambda e: e['timestamp'])
            
            return order_events


class SqliteAuditLog(TradeAuditLog):
    """
    Stores audit events in a SQLite database
    
    Provides faster querying capabilities compared to JSON files.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the SQLite audit log
        
        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__()
        self.db_path = db_path
        
        # Create database and tables if they don't exist
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    broker_id TEXT,
                    order_id TEXT,
                    strategy_id TEXT,
                    details TEXT NOT NULL
                )
            ''')
            
            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_id ON events (order_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_broker_id ON events (broker_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events (event_type)')
            
            conn.commit()
        
        logger.info(f"Initialized SqliteAuditLog with database: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        return conn
    
    def log_event(self, event_type: AuditEventType, details: Dict[str, Any],
                broker_id: Optional[str] = None, order_id: Optional[str] = None,
                strategy_id: Optional[str] = None) -> str:
        """Log an event to the audit trail"""
        with self._lock:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Convert details to JSON
            details_json = json.dumps(details)
            
            # Insert into database
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO events
                    (event_id, timestamp, event_type, broker_id, order_id, strategy_id, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_id,
                    datetime.now().isoformat(),
                    event_type.value,
                    broker_id,
                    order_id,
                    strategy_id,
                    details_json
                ))
                
                conn.commit()
            
            return event_id
    
    def get_events(self, filters: Optional[Dict[str, Any]] = None,
                limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve events from the audit log"""
        # Default filters
        if filters is None:
            filters = {}
        
        # Build query
        query = "SELECT * FROM events"
        params = []
        
        where_clauses = []
        
        if 'event_type' in filters:
            where_clauses.append("event_type = ?")
            params.append(filters['event_type'])
        
        if 'broker_id' in filters:
            where_clauses.append("broker_id = ?")
            params.append(filters['broker_id'])
        
        if 'order_id' in filters:
            where_clauses.append("order_id = ?")
            params.append(filters['order_id'])
        
        if 'strategy_id' in filters:
            where_clauses.append("strategy_id = ?")
            params.append(filters['strategy_id'])
        
        if 'start_date' in filters:
            where_clauses.append("timestamp >= ?")
            params.append(filters['start_date'])
        
        if 'end_date' in filters:
            where_clauses.append("timestamp <= ?")
            params.append(filters['end_date'])
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Add ordering
        query += " ORDER BY timestamp DESC"
        
        # Add pagination
        query += " LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)
        
        # Execute query
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                event = {
                    'event_id': row[0],
                    'timestamp': row[1],
                    'event_type': row[2],
                    'details': json.loads(row[6])
                }
                
                if row[3]:  # broker_id
                    event['broker_id'] = row[3]
                
                if row[4]:  # order_id
                    event['order_id'] = row[4]
                
                if row[5]:  # strategy_id
                    event['strategy_id'] = row[5]
                
                events.append(event)
            
            return events
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific event by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            
            if row:
                event = {
                    'event_id': row[0],
                    'timestamp': row[1],
                    'event_type': row[2],
                    'details': json.loads(row[6])
                }
                
                if row[3]:  # broker_id
                    event['broker_id'] = row[3]
                
                if row[4]:  # order_id
                    event['order_id'] = row[4]
                
                if row[5]:  # strategy_id
                    event['strategy_id'] = row[5]
                
                return event
            
            return None
    
    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """Retrieve the complete history for an order"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM events WHERE order_id = ? ORDER BY timestamp",
                (order_id,)
            )
            
            events = []
            for row in cursor.fetchall():
                event = {
                    'event_id': row[0],
                    'timestamp': row[1],
                    'event_type': row[2],
                    'details': json.loads(row[6])
                }
                
                if row[3]:  # broker_id
                    event['broker_id'] = row[3]
                
                if row[4]:  # order_id
                    event['order_id'] = row[4]
                
                if row[5]:  # strategy_id
                    event['strategy_id'] = row[5]
                
                events.append(event)
            
            return events


class AuditLogFactory:
    """
    Factory for creating audit log instances
    
    Provides methods to create different types of audit logs.
    """
    
    @staticmethod
    def create_json_file_log(log_dir: str) -> JsonFileAuditLog:
        """
        Create a JSON file audit log
        
        Args:
            log_dir: Directory for log files
            
        Returns:
            JsonFileAuditLog: Audit log instance
        """
        return JsonFileAuditLog(log_dir)
    
    @staticmethod
    def create_sqlite_log(db_path: str) -> SqliteAuditLog:
        """
        Create a SQLite audit log
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            SqliteAuditLog: Audit log instance
        """
        return SqliteAuditLog(db_path)
