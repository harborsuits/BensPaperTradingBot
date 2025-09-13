#!/usr/bin/env python3
"""
Broker Metrics System - Base Components

Defines the core abstractions for collecting and analyzing broker performance metrics.
"""

import time
import json
import logging
import threading
import statistics
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be collected"""
    LATENCY = "latency"
    RELIABILITY = "reliability"
    EXECUTION_QUALITY = "execution_quality"
    COST = "cost"
    HEALTH = "health"

class MetricOperation(Enum):
    """Operations that generate metrics"""
    # Account operations
    GET_ACCOUNT = "get_account"
    GET_POSITIONS = "get_positions"
    
    # Order operations
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    GET_ORDERS = "get_orders"
    GET_ORDER_STATUS = "get_order_status"
    
    # Market data operations
    GET_QUOTE = "get_quote"
    GET_CANDLES = "get_candles"
    
    # Connection operations
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"
    
    # General operations
    API_CALL = "api_call"
    ERROR = "error"
    HEALTH_CHECK = "health_check"

class MetricPeriod(Enum):
    """Time periods for metric aggregation"""
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1mo"

class MetricValue:
    """Represents a metric value with timestamp and metadata"""
    
    def __init__(
        self,
        value: Union[float, int, bool, str],
        metric_type: MetricType,
        operation: MetricOperation,
        broker_id: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a metric value
        
        Args:
            value: The metric value
            metric_type: Type of metric
            operation: Operation that generated the metric
            broker_id: Broker identifier
            timestamp: Timestamp in seconds (default: now)
            metadata: Additional contextual data
        """
        self.value = value
        self.metric_type = metric_type
        self.operation = operation
        self.broker_id = broker_id
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "value": self.value,
            "metric_type": self.metric_type.value,
            "operation": self.operation.value,
            "broker_id": self.broker_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricValue':
        """Create from dictionary"""
        return cls(
            value=data["value"],
            metric_type=MetricType(data["metric_type"]),
            operation=MetricOperation(data["operation"]),
            broker_id=data["broker_id"],
            timestamp=data["timestamp"],
            metadata=data["metadata"]
        )

class MetricWindow:
    """A time window of metric values with statistical aggregation"""
    
    def __init__(
        self,
        period: MetricPeriod,
        capacity: int = 1000,
        expire_after: Optional[int] = None
    ):
        """
        Initialize a metric window
        
        Args:
            period: Time period for this window
            capacity: Maximum number of values to store
            expire_after: Time in seconds after which to expire values
        """
        self.period = period
        self.capacity = capacity
        self.expire_after = expire_after
        self.values = deque(maxlen=capacity)
        self.last_cleanup = time.time()
    
    def add(self, value: MetricValue):
        """Add a value to the window"""
        self.values.append(value)
        
        # Periodically clean up expired values
        if self.expire_after and time.time() - self.last_cleanup > 60:
            self._cleanup()
    
    def _cleanup(self):
        """Remove expired values"""
        if not self.expire_after:
            return
            
        cutoff = time.time() - self.expire_after
        self.values = deque([v for v in self.values if v.timestamp >= cutoff], maxlen=self.capacity)
        self.last_cleanup = time.time()
    
    def get_values(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[MetricValue]:
        """Get values matching the criteria"""
        filtered = self.values
        
        if metric_type:
            filtered = [v for v in filtered if v.metric_type == metric_type]
        
        if operation:
            filtered = [v for v in filtered if v.operation == operation]
        
        if broker_id:
            filtered = [v for v in filtered if v.broker_id == broker_id]
        
        if start_time:
            filtered = [v for v in filtered if v.timestamp >= start_time]
        
        if end_time:
            filtered = [v for v in filtered if v.timestamp <= end_time]
        
        return list(filtered)
    
    def get_raw_values(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None
    ) -> List[Union[float, int, bool, str]]:
        """Get raw values (not MetricValue objects) matching the criteria"""
        values = self.get_values(metric_type, operation, broker_id)
        return [v.value for v in values if isinstance(v.value, (float, int, bool, str))]
    
    def get_stats(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistical summary of values"""
        values = self.get_raw_values(metric_type, operation, broker_id)
        
        # Filter to numeric values for statistics
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "stddev": None
            }
        
        try:
            stats = {
                "count": len(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": statistics.mean(numeric_values),
                "median": statistics.median(numeric_values),
                "stddev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
            }
        except statistics.StatisticsError:
            stats = {
                "count": len(numeric_values),
                "min": min(numeric_values) if numeric_values else None,
                "max": max(numeric_values) if numeric_values else None,
                "mean": sum(numeric_values) / len(numeric_values) if numeric_values else None,
                "median": sorted(numeric_values)[len(numeric_values) // 2] if numeric_values else None,
                "stddev": None
            }
        
        return stats

class MetricsStore(ABC):
    """Abstract base class for metrics storage"""
    
    @abstractmethod
    def store(self, metric: MetricValue):
        """Store a metric value"""
        pass
    
    @abstractmethod
    def query(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[MetricValue]:
        """Query stored metrics"""
        pass
    
    @abstractmethod
    def get_stats(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        period: MetricPeriod = MetricPeriod.DAY
    ) -> Dict[str, Any]:
        """Get statistical summary of metrics"""
        pass

class InMemoryMetricsStore(MetricsStore):
    """In-memory implementation of metrics storage"""
    
    def __init__(self, max_retention_days: int = 7):
        """
        Initialize in-memory metrics store
        
        Args:
            max_retention_days: Maximum number of days to retain metrics
        """
        self.max_retention_days = max_retention_days
        self.windows = {
            MetricPeriod.MINUTE: MetricWindow(MetricPeriod.MINUTE, expire_after=60*60),  # 1 hour
            MetricPeriod.FIVE_MINUTES: MetricWindow(MetricPeriod.FIVE_MINUTES, expire_after=60*60*6),  # 6 hours
            MetricPeriod.FIFTEEN_MINUTES: MetricWindow(MetricPeriod.FIFTEEN_MINUTES, expire_after=60*60*24),  # 1 day
            MetricPeriod.HOUR: MetricWindow(MetricPeriod.HOUR, expire_after=60*60*24*2),  # 2 days
            MetricPeriod.DAY: MetricWindow(MetricPeriod.DAY, expire_after=60*60*24*max_retention_days)  # N days
        }
        self.lock = threading.RLock()
    
    def store(self, metric: MetricValue):
        """Store a metric value in all time windows"""
        with self.lock:
            for window in self.windows.values():
                window.add(metric)
    
    def query(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000,
        period: MetricPeriod = MetricPeriod.MINUTE
    ) -> List[MetricValue]:
        """Query stored metrics from a specific window"""
        with self.lock:
            if period not in self.windows:
                raise ValueError(f"Invalid period: {period}")
            
            window = self.windows[period]
            values = window.get_values(
                metric_type=metric_type,
                operation=operation,
                broker_id=broker_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Sort by timestamp (most recent first) and apply limit
            values.sort(key=lambda x: x.timestamp, reverse=True)
            return values[:limit]
    
    def get_stats(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        period: MetricPeriod = MetricPeriod.DAY
    ) -> Dict[str, Any]:
        """Get statistical summary from a specific window"""
        with self.lock:
            if period not in self.windows:
                raise ValueError(f"Invalid period: {period}")
            
            window = self.windows[period]
            return window.get_stats(
                metric_type=metric_type,
                operation=operation,
                broker_id=broker_id
            )

class FileMetricsStore(MetricsStore):
    """File-based implementation of metrics storage"""
    
    def __init__(self, file_path: str, flush_interval: int = 60):
        """
        Initialize file-based metrics store
        
        Args:
            file_path: Path to metrics file
            flush_interval: How often to flush to disk (seconds)
        """
        self.file_path = file_path
        self.flush_interval = flush_interval
        self.in_memory = InMemoryMetricsStore()
        self.last_flush = time.time()
        self.lock = threading.RLock()
        self.metrics_to_flush = []
        
        # Create file if it doesn't exist
        with open(self.file_path, 'a') as f:
            pass
        
        # Load existing metrics
        self._load_from_file()
    
    def _load_from_file(self):
        """Load metrics from file"""
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        metric = MetricValue.from_dict(data)
                        self.in_memory.store(metric)
                    except Exception as e:
                        logger.error(f"Error parsing metric from file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading metrics from file: {str(e)}")
    
    def _flush_to_file(self):
        """Flush pending metrics to file"""
        if not self.metrics_to_flush:
            return
            
        try:
            with open(self.file_path, 'a') as f:
                for metric in self.metrics_to_flush:
                    f.write(json.dumps(metric.to_dict()) + "\n")
            
            self.metrics_to_flush = []
            self.last_flush = time.time()
        except Exception as e:
            logger.error(f"Error flushing metrics to file: {str(e)}")
    
    def store(self, metric: MetricValue):
        """Store a metric value"""
        with self.lock:
            # Store in memory
            self.in_memory.store(metric)
            
            # Queue for file storage
            self.metrics_to_flush.append(metric)
            
            # Flush if interval has elapsed
            if time.time() - self.last_flush > self.flush_interval:
                self._flush_to_file()
    
    def query(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000,
        period: MetricPeriod = MetricPeriod.MINUTE
    ) -> List[MetricValue]:
        """Query stored metrics"""
        with self.lock:
            # Flush pending metrics to ensure all data is in memory
            self._flush_to_file()
            
            # Query in-memory store
            return self.in_memory.query(
                metric_type=metric_type,
                operation=operation,
                broker_id=broker_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                period=period
            )
    
    def get_stats(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[MetricOperation] = None,
        broker_id: Optional[str] = None,
        period: MetricPeriod = MetricPeriod.DAY
    ) -> Dict[str, Any]:
        """Get statistical summary"""
        with self.lock:
            # Flush pending metrics to ensure all data is in memory
            self._flush_to_file()
            
            # Get stats from in-memory store
            return self.in_memory.get_stats(
                metric_type=metric_type,
                operation=operation,
                broker_id=broker_id,
                period=period
            )
