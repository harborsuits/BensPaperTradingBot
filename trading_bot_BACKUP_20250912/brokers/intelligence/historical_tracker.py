#!/usr/bin/env python3
"""
Broker Performance Historical Tracker

Records and analyzes historical broker performance metrics over time.
Provides time-series storage, trend analysis, anomaly detection,
and data access for visualizations and predictive modeling.
"""

import os
import csv
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from threading import Lock, Thread
from time import sleep

from trading_bot.event_system.event_bus import EventBus
from trading_bot.event_system.event_types import EventType
from trading_bot.brokers.metrics.base import MetricType, MetricOperation


logger = logging.getLogger(__name__)


class BrokerPerformanceRecord:
    """Individual performance record at a point in time"""
    
    def __init__(
        self, 
        broker_id: str, 
        timestamp: datetime,
        metrics: Dict[str, Any],
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None
    ):
        """
        Initialize a broker performance record
        
        Args:
            broker_id: Broker identifier
            timestamp: Record timestamp
            metrics: Raw metrics data
            asset_class: Optional asset class context
            operation_type: Optional operation type context
            scores: Optional performance scores
        """
        self.broker_id = broker_id
        self.timestamp = timestamp
        self.metrics = metrics
        self.asset_class = asset_class
        self.operation_type = operation_type
        self.scores = scores or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "broker_id": self.broker_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "asset_class": self.asset_class,
            "operation_type": self.operation_type,
            "scores": self.scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokerPerformanceRecord':
        """Create from dictionary"""
        return cls(
            broker_id=data["broker_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=data["metrics"],
            asset_class=data.get("asset_class"),
            operation_type=data.get("operation_type"),
            scores=data.get("scores")
        )
    
    def get_flat_metrics(self) -> Dict[str, float]:
        """
        Get flattened metrics dictionary for storage
        
        Returns:
            Dict with metrics in format {category_name: value}
        """
        flat_metrics = {}
        
        # Process each metric category
        for category, values in self.metrics.items():
            if isinstance(values, dict):
                # Flatten nested dictionaries
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[f"{category}_{key}"] = value
            elif isinstance(values, (int, float)):
                # Direct values
                flat_metrics[category] = values
        
        # Add scores if present
        for score_name, score_value in self.scores.items():
            flat_metrics[f"score_{score_name}"] = score_value
        
        return flat_metrics


class SQLiteTimeSeriesStore:
    """SQLite-based time series storage for broker performance data"""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite storage
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.lock = Lock()
        
        # Initialize database
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS broker_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                broker_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                asset_class TEXT,
                operation_type TEXT,
                metrics_json TEXT NOT NULL,
                scores_json TEXT
            )
            """)
            
            # Create indexes for efficient queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_broker_timestamp ON broker_metrics (broker_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_asset_class ON broker_metrics (asset_class)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_operation_type ON broker_metrics (operation_type)")
            
            # Create metrics columns table to track schema
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_columns (
                column_name TEXT PRIMARY KEY,
                data_type TEXT NOT NULL
            )
            """)
            
            conn.commit()
    
    def _get_connection(self):
        """Get database connection"""
        if self.connection is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Create connection
            self.connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                isolation_level=None  # autocommit mode
            )
            
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Configure connection
            self.connection.row_factory = sqlite3.Row
        
        return self.connection
    
    def close(self):
        """Close database connection"""
        with self.lock:
            if self.connection:
                self.connection.close()
                self.connection = None
    
    def store_record(self, record: BrokerPerformanceRecord) -> bool:
        """
        Store a broker performance record
        
        Args:
            record: Performance record to store
            
        Returns:
            bool: True if stored successfully
        """
        try:
            with self.lock, self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Store base record
                cursor.execute(
                    """
                    INSERT INTO broker_metrics 
                    (broker_id, timestamp, asset_class, operation_type, metrics_json, scores_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.broker_id,
                        record.timestamp,
                        record.asset_class,
                        record.operation_type,
                        json.dumps(record.metrics),
                        json.dumps(record.scores) if record.scores else None
                    )
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store record: {str(e)}")
            return False
    
    def get_records(
        self,
        broker_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[BrokerPerformanceRecord]:
        """
        Retrieve broker performance records
        
        Args:
            broker_id: Optional broker ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            limit: Maximum number of records to return
            
        Returns:
            List of BrokerPerformanceRecord objects
        """
        query = "SELECT * FROM broker_metrics WHERE 1=1"
        params = []
        
        # Add filters
        if broker_id:
            query += " AND broker_id = ?"
            params.append(broker_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if asset_class:
            query += " AND asset_class = ?"
            params.append(asset_class)
        
        if operation_type:
            query += " AND operation_type = ?"
            params.append(operation_type)
        
        # Add order and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with self.lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append(BrokerPerformanceRecord(
                        broker_id=row["broker_id"],
                        timestamp=row["timestamp"],
                        metrics=json.loads(row["metrics_json"]),
                        asset_class=row["asset_class"],
                        operation_type=row["operation_type"],
                        scores=json.loads(row["scores_json"]) if row["scores_json"] else None
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get records: {str(e)}")
            return []
    
    def get_as_dataframe(
        self,
        broker_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Get records as pandas DataFrame for analysis
        
        Args:
            broker_id: Optional broker ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with broker performance data
        """
        records = self.get_records(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type,
            limit=limit
        )
        
        if not records:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'broker_id', 'timestamp', 'asset_class', 'operation_type'
            ])
        
        # Start with basic fields
        data = []
        for record in records:
            row = {
                'broker_id': record.broker_id,
                'timestamp': record.timestamp,
                'asset_class': record.asset_class,
                'operation_type': record.operation_type
            }
            
            # Add flattened metrics
            row.update(record.get_flat_metrics())
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def prune_old_records(self, max_age_days: int = 90) -> int:
        """
        Remove records older than specified age
        
        Args:
            max_age_days: Maximum age in days to keep
            
        Returns:
            int: Number of records removed
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        try:
            with self.lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM broker_metrics WHERE timestamp < ?",
                    (cutoff_date,)
                )
                
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Failed to prune old records: {str(e)}")
            return 0


class CSVTimeSeriesStore:
    """CSV-based time series storage for broker performance data"""
    
    def __init__(self, base_dir: str):
        """
        Initialize CSV storage
        
        Args:
            base_dir: Base directory for CSV files
        """
        self.base_dir = base_dir
        self.lock = Lock()
        
        # Ensure directory exists
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _get_broker_file_path(self, broker_id: str, date: datetime) -> str:
        """Get file path for broker and date"""
        # Create directory structure
        year_dir = os.path.join(self.base_dir, str(date.year))
        month_dir = os.path.join(year_dir, f"{date.month:02d}")
        os.makedirs(month_dir, exist_ok=True)
        
        # File naming includes broker ID and date
        file_name = f"{broker_id}_{date.strftime('%Y-%m-%d')}.csv"
        return os.path.join(month_dir, file_name)
    
    def _ensure_file_headers(self, file_path: str, record: BrokerPerformanceRecord) -> List[str]:
        """
        Ensure CSV file has appropriate headers
        
        Args:
            file_path: Path to CSV file
            record: Record to determine headers from
            
        Returns:
            List of header columns
        """
        # Basic fields
        headers = [
            'broker_id', 'timestamp', 'asset_class', 
            'operation_type'
        ]
        
        # Add metrics fields
        flat_metrics = record.get_flat_metrics()
        metric_headers = list(flat_metrics.keys())
        headers.extend(metric_headers)
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        
        if not file_exists:
            # Create new file with headers
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        return headers
    
    def store_record(self, record: BrokerPerformanceRecord) -> bool:
        """
        Store a broker performance record
        
        Args:
            record: Performance record to store
            
        Returns:
            bool: True if stored successfully
        """
        try:
            with self.lock:
                # Get file path for this record
                file_path = self._get_broker_file_path(
                    record.broker_id, 
                    record.timestamp
                )
                
                # Ensure headers exist
                headers = self._ensure_file_headers(file_path, record)
                
                # Prepare row data
                row_data = {
                    'broker_id': record.broker_id,
                    'timestamp': record.timestamp.isoformat(),
                    'asset_class': record.asset_class,
                    'operation_type': record.operation_type
                }
                
                # Add flattened metrics
                row_data.update(record.get_flat_metrics())
                
                # Write to CSV
                with open(file_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                    writer.writerow(row_data)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store record: {str(e)}")
            return False
    
    def get_records(
        self,
        broker_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[BrokerPerformanceRecord]:
        """
        Retrieve broker performance records
        
        Args:
            broker_id: Optional broker ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            limit: Maximum number of records to return
            
        Returns:
            List of BrokerPerformanceRecord objects
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        
        if not end_time:
            end_time = datetime.now()
        
        results = []
        
        try:
            with self.lock:
                # Find relevant files based on date range
                current_date = start_time.date()
                end_date = end_time.date()
                
                while current_date <= end_date:
                    # Get file paths for this date
                    file_paths = []
                    
                    if broker_id:
                        # Specific broker
                        file_path = self._get_broker_file_path(broker_id, current_date)
                        if os.path.exists(file_path):
                            file_paths.append(file_path)
                    else:
                        # All brokers - find all matching files for this date
                        year_dir = os.path.join(self.base_dir, str(current_date.year))
                        month_dir = os.path.join(year_dir, f"{current_date.month:02d}")
                        if os.path.exists(month_dir):
                            date_pattern = current_date.strftime('%Y-%m-%d')
                            for file_name in os.listdir(month_dir):
                                if date_pattern in file_name:
                                    file_paths.append(os.path.join(month_dir, file_name))
                    
                    # Process files
                    for file_path in file_paths:
                        if not os.path.exists(file_path):
                            continue
                        
                        # Read CSV
                        df = pd.read_csv(file_path)
                        
                        # Apply filters
                        if asset_class:
                            df = df[df['asset_class'] == asset_class]
                        
                        if operation_type:
                            df = df[df['operation_type'] == operation_type]
                        
                        # Apply time filters
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                        
                        # Convert to records
                        for _, row in df.iterrows():
                            # Extract metrics and scores
                            metrics = {}
                            scores = {}
                            
                            for col, value in row.items():
                                if col.startswith('score_'):
                                    scores[col[6:]] = value
                                elif col not in ['broker_id', 'timestamp', 'asset_class', 'operation_type']:
                                    # Parse metric category
                                    if '_' in col:
                                        category, name = col.split('_', 1)
                                        if category not in metrics:
                                            metrics[category] = {}
                                        metrics[category][name] = value
                                    else:
                                        metrics[col] = value
                            
                            # Create record
                            record = BrokerPerformanceRecord(
                                broker_id=row['broker_id'],
                                timestamp=row['timestamp'],
                                metrics=metrics,
                                asset_class=row.get('asset_class'),
                                operation_type=row.get('operation_type'),
                                scores=scores if scores else None
                            )
                            
                            results.append(record)
                            
                            # Check limit
                            if len(results) >= limit:
                                return results[:limit]
                    
                    # Move to next date
                    current_date += timedelta(days=1)
                
                # Sort by timestamp
                results.sort(key=lambda r: r.timestamp, reverse=True)
                
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get records: {str(e)}")
            return []
