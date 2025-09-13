"""
Broker Health Metrics

Provides tracking and analysis of broker performance and reliability metrics
to enable smart routing decisions and degradation detection.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
from collections import deque
import threading
import math
import statistics

from trading_bot.event_system import EventBus, Event

# Configure logging
logger = logging.getLogger(__name__)

class BrokerHealthMetrics:
    """
    Tracks health and performance metrics for broker connections
    
    Features:
    - Success/failure rate tracking
    - Response time analysis
    - Adaptive timeout calculation
    - Degradation detection
    - Pattern analysis for intermittent failures
    """
    
    def __init__(self, broker_id: str, window_size: int = 100, degradation_threshold: float = 0.3):
        """
        Initialize health metrics tracker for a specific broker
        
        Args:
            broker_id: Unique identifier for the broker
            window_size: Number of operations to track in sliding window
            degradation_threshold: Failure rate threshold to mark as degraded (0.0-1.0)
        """
        self.broker_id = broker_id
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        
        # Operation tracking
        self._operations = deque(maxlen=window_size)
        self._response_times = deque(maxlen=window_size)
        self._failure_timestamps = deque(maxlen=window_size)
        
        # Current state
        self._degraded = False
        self._timeout_multiplier = 1.0
        self._baseline_timeout = 10.0  # Default timeout in seconds
        self._calculated_timeout = self._baseline_timeout
        
        # For thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._total_operations = 0
        self._total_failures = 0
        self._last_reset = datetime.now()
        
        # Failure patterns
        self._failure_pattern_detected = False
        self._pattern_description = None
        
        # Event bus for notifications
        self.event_bus = EventBus()
        
        logger.info(f"Initialized health metrics for broker '{broker_id}'")
    
    def record_operation(self, operation_type: str, success: bool, response_time: float,
                      error_message: Optional[str] = None) -> None:
        """
        Record the outcome of a broker operation
        
        Args:
            operation_type: Type of operation (e.g., 'place_order', 'get_quote')
            success: Whether the operation succeeded
            response_time: Time taken to complete the operation in seconds
            error_message: Error message if operation failed
        """
        with self._lock:
            timestamp = datetime.now()
            
            operation_data = {
                'timestamp': timestamp,
                'type': operation_type,
                'success': success,
                'response_time': response_time,
                'error_message': error_message
            }
            
            self._operations.append(operation_data)
            self._response_times.append(response_time)
            
            if not success:
                self._failure_timestamps.append(timestamp)
                self._total_failures += 1
                
                # Check for degradation after failure
                self._check_degradation()
                
                # Check for failure patterns
                self._analyze_failure_patterns()
                
                # Update timeout based on recent performance
                self._update_timeout()
            
            self._total_operations += 1
            
            # Publish metrics update event
            failure_rate = self.get_failure_rate()
            avg_response_time = self.get_average_response_time()
            
            self.event_bus.publish(Event(
                "broker_health_updated",
                {
                    "broker_id": self.broker_id,
                    "timestamp": timestamp.isoformat(),
                    "operation_type": operation_type,
                    "success": success,
                    "failure_rate": failure_rate,
                    "avg_response_time": avg_response_time,
                    "timeout": self._calculated_timeout,
                    "degraded": self._degraded
                }
            ))
    
    def get_failure_rate(self) -> float:
        """
        Get the current failure rate based on recent operations
        
        Returns:
            float: Failure rate as a proportion (0.0-1.0)
        """
        with self._lock:
            if not self._operations:
                return 0.0
            
            failures = sum(1 for op in self._operations if not op['success'])
            return failures / len(self._operations)
    
    def get_average_response_time(self) -> float:
        """
        Get the average response time of recent operations
        
        Returns:
            float: Average response time in seconds
        """
        with self._lock:
            if not self._response_times:
                return 0.0
            
            return sum(self._response_times) / len(self._response_times)
    
    def get_timeout(self) -> float:
        """
        Get the recommended timeout for this broker based on performance
        
        Returns:
            float: Recommended timeout in seconds
        """
        with self._lock:
            return self._calculated_timeout
    
    def is_degraded(self) -> bool:
        """
        Check if the broker is currently considered degraded
        
        Returns:
            bool: True if broker is in degraded state
        """
        with self._lock:
            return self._degraded
    
    def has_failure_pattern(self) -> bool:
        """
        Check if a failure pattern has been detected
        
        Returns:
            bool: True if a pattern of failures has been detected
        """
        with self._lock:
            return self._failure_pattern_detected
    
    def get_pattern_description(self) -> Optional[str]:
        """
        Get description of detected failure pattern
        
        Returns:
            Optional[str]: Description of the pattern or None if no pattern
        """
        with self._lock:
            return self._pattern_description
    
    def reset_metrics(self) -> None:
        """
        Reset all metrics and tracking data
        """
        with self._lock:
            self._operations.clear()
            self._response_times.clear()
            self._failure_timestamps.clear()
            
            self._degraded = False
            self._timeout_multiplier = 1.0
            self._calculated_timeout = self._baseline_timeout
            
            self._total_operations = 0
            self._total_failures = 0
            self._last_reset = datetime.now()
            
            self._failure_pattern_detected = False
            self._pattern_description = None
            
            logger.info(f"Reset health metrics for broker '{self.broker_id}'")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current health metrics
        
        Returns:
            Dict: Summary of broker health metrics
        """
        with self._lock:
            return {
                'broker_id': self.broker_id,
                'total_operations': self._total_operations,
                'total_failures': self._total_failures,
                'failure_rate': self.get_failure_rate(),
                'avg_response_time': self.get_average_response_time(),
                'timeout': self._calculated_timeout,
                'degraded': self._degraded,
                'has_failure_pattern': self._failure_pattern_detected,
                'pattern_description': self._pattern_description,
                'metrics_since': self._last_reset.isoformat()
            }
    
    def _check_degradation(self) -> None:
        """Check if broker should be marked as degraded based on recent performance"""
        failure_rate = self.get_failure_rate()
        
        # Mark as degraded if failure rate exceeds threshold
        if failure_rate >= self.degradation_threshold:
            if not self._degraded:
                logger.warning(f"Broker '{self.broker_id}' marked as DEGRADED (failure rate: {failure_rate:.2f})")
                self._degraded = True
                
                # Publish degradation event
                self.event_bus.publish(Event(
                    "broker_degraded",
                    {
                        "broker_id": self.broker_id,
                        "timestamp": datetime.now().isoformat(),
                        "failure_rate": failure_rate,
                        "threshold": self.degradation_threshold
                    }
                ))
        elif failure_rate < self.degradation_threshold / 2:  # Hysteresis to prevent flapping
            if self._degraded:
                logger.info(f"Broker '{self.broker_id}' recovered from DEGRADED state (failure rate: {failure_rate:.2f})")
                self._degraded = False
                
                # Publish recovery event
                self.event_bus.publish(Event(
                    "broker_recovered",
                    {
                        "broker_id": self.broker_id,
                        "timestamp": datetime.now().isoformat(),
                        "failure_rate": failure_rate
                    }
                ))
    
    def _update_timeout(self) -> None:
        """Update timeout value based on recent performance using exponential backoff"""
        if not self._response_times:
            self._calculated_timeout = self._baseline_timeout
            return
        
        # Calculate baseline timeout as 95th percentile of response times
        if len(self._response_times) >= 20:
            response_times_list = list(self._response_times)
            percentile_95 = statistics.quantiles(response_times_list, n=20)[19]  # 95th percentile
            self._baseline_timeout = max(1.0, percentile_95 * 2)  # Double the 95th percentile, min 1 second
        
        # Calculate failure rate over recent operations
        failure_rate = self.get_failure_rate()
        
        # Calculate timeout multiplier using exponential backoff based on failure rate
        if failure_rate > 0:
            # Exponential backoff that increases with failure rate
            # At threshold, multiplier is about 4x
            # At 50% failure, multiplier is about 8x
            backoff_exponent = min(5, -math.log(1 - min(0.9, failure_rate)) * 10)
            self._timeout_multiplier = max(1.0, backoff_exponent)
        else:
            # No failures, gradually decrease timeout multiplier
            self._timeout_multiplier = max(1.0, self._timeout_multiplier * 0.9)
        
        # Set calculated timeout
        self._calculated_timeout = self._baseline_timeout * self._timeout_multiplier
        
        # Cap at reasonable maximum (2 minutes)
        self._calculated_timeout = min(120.0, self._calculated_timeout)
    
    def _analyze_failure_patterns(self) -> None:
        """Analyze timestamps of failures to detect patterns"""
        if len(self._failure_timestamps) < 5:
            self._failure_pattern_detected = False
            self._pattern_description = None
            return
        
        # Check for periodic failures
        if self._detect_periodic_failures():
            return
        
        # Check for time-of-day patterns
        if self._detect_time_of_day_pattern():
            return
        
        # Check for consecutive failures
        if self._detect_consecutive_failures():
            return
        
        # No pattern detected
        self._failure_pattern_detected = False
        self._pattern_description = None
    
    def _detect_periodic_failures(self) -> bool:
        """Detect periodic failure patterns"""
        if len(self._failure_timestamps) < 5:
            return False
        
        # Convert to list and sort
        timestamps = sorted(list(self._failure_timestamps))
        
        # Calculate intervals between consecutive failures
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        # Check if intervals are similar (within 10% of each other)
        if len(intervals) >= 3:
            avg_interval = sum(intervals) / len(intervals)
            similar_intervals = sum(1 for i in intervals if abs(i - avg_interval) / avg_interval < 0.1)
            
            if similar_intervals >= min(3, len(intervals) * 0.7):
                self._failure_pattern_detected = True
                self._pattern_description = f"Periodic failures every {avg_interval:.1f} seconds"
                logger.warning(f"Broker '{self.broker_id}': {self._pattern_description}")
                return True
        
        return False
    
    def _detect_time_of_day_pattern(self) -> bool:
        """Detect time-of-day patterns in failures"""
        if len(self._failure_timestamps) < 5:
            return False
        
        # Extract hours from timestamps
        hours = [ts.hour for ts in self._failure_timestamps]
        
        # Count failures by hour
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find peak hours (where at least 30% of failures occur)
        total_failures = len(self._failure_timestamps)
        peak_hours = [hour for hour, count in hour_counts.items() 
                     if count >= total_failures * 0.3]
        
        if peak_hours:
            peak_hours_str = ", ".join([f"{h}:00" for h in sorted(peak_hours)])
            self._failure_pattern_detected = True
            self._pattern_description = f"Time-of-day pattern: {peak_hours_str}"
            logger.warning(f"Broker '{self.broker_id}': {self._pattern_description}")
            return True
        
        return False
    
    def _detect_consecutive_failures(self) -> bool:
        """Detect consecutive failure patterns"""
        if len(self._failure_timestamps) < 3:
            return False
        
        # Convert to list and sort
        timestamps = sorted(list(self._failure_timestamps))
        
        # Look for sequences of failures close together (within 1 second)
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i-1]).total_seconds() <= 1.0:
                consecutive_count += 1
            else:
                consecutive_count = 1
            
            max_consecutive = max(max_consecutive, consecutive_count)
        
        if max_consecutive >= 3:
            self._failure_pattern_detected = True
            self._pattern_description = f"Consecutive failures ({max_consecutive} in a row)"
            logger.warning(f"Broker '{self.broker_id}': {self._pattern_description}")
            return True
        
        return False


class BrokerHealthManager:
    """
    Manages health metrics for multiple brokers and provides
    decision support for smart routing and failover
    """
    
    def __init__(self):
        """Initialize the broker health manager"""
        self.brokers = {}  # broker_id -> BrokerHealthMetrics
        self._lock = threading.RLock()
        self.event_bus = EventBus()
        
        logger.info("Initialized BrokerHealthManager")
    
    def register_broker(self, broker_id: str, window_size: int = 100, 
                      degradation_threshold: float = 0.3) -> None:
        """
        Register a broker for health tracking
        
        Args:
            broker_id: Unique identifier for the broker
            window_size: Number of operations to track in sliding window
            degradation_threshold: Failure rate threshold to mark as degraded (0.0-1.0)
        """
        with self._lock:
            if broker_id in self.brokers:
                logger.warning(f"Broker '{broker_id}' already registered for health tracking, replacing")
            
            self.brokers[broker_id] = BrokerHealthMetrics(
                broker_id, 
                window_size=window_size,
                degradation_threshold=degradation_threshold
            )
            
            logger.info(f"Registered broker '{broker_id}' for health tracking")
    
    def record_operation(self, broker_id: str, operation_type: str, success: bool, 
                       response_time: float, error_message: Optional[str] = None) -> None:
        """
        Record a broker operation outcome
        
        Args:
            broker_id: Unique identifier for the broker
            operation_type: Type of operation (e.g., 'place_order', 'get_quote')
            success: Whether the operation succeeded
            response_time: Time taken to complete the operation in seconds
            error_message: Error message if operation failed
        """
        with self._lock:
            if broker_id not in self.brokers:
                logger.warning(f"Broker '{broker_id}' not registered for health tracking, registering now")
                self.register_broker(broker_id)
            
            self.brokers[broker_id].record_operation(
                operation_type, success, response_time, error_message
            )
    
    def is_broker_healthy(self, broker_id: str) -> bool:
        """
        Check if a broker is considered healthy
        
        Args:
            broker_id: Unique identifier for the broker
            
        Returns:
            bool: True if broker is healthy, False if degraded or not registered
        """
        with self._lock:
            if broker_id not in self.brokers:
                # Unregistered brokers are considered unhealthy
                return False
            
            return not self.brokers[broker_id].is_degraded()
    
    def get_broker_timeout(self, broker_id: str, default_timeout: float = 10.0) -> float:
        """
        Get recommended timeout for a broker
        
        Args:
            broker_id: Unique identifier for the broker
            default_timeout: Default timeout to use if broker not registered
            
        Returns:
            float: Recommended timeout in seconds
        """
        with self._lock:
            if broker_id not in self.brokers:
                return default_timeout
            
            return self.brokers[broker_id].get_timeout()
    
    def get_healthy_brokers(self) -> List[str]:
        """
        Get list of healthy broker IDs
        
        Returns:
            List[str]: List of broker IDs considered healthy
        """
        with self._lock:
            return [
                broker_id for broker_id, metrics in self.brokers.items()
                if not metrics.is_degraded()
            ]
    
    def get_all_broker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health metrics summary for all brokers
        
        Returns:
            Dict: Map of broker IDs to their metric summaries
        """
        with self._lock:
            return {
                broker_id: metrics.get_metrics_summary()
                for broker_id, metrics in self.brokers.items()
            }
    
    def reset_broker_metrics(self, broker_id: str) -> bool:
        """
        Reset health metrics for a specific broker
        
        Args:
            broker_id: Unique identifier for the broker
            
        Returns:
            bool: True if broker was found and reset, False otherwise
        """
        with self._lock:
            if broker_id not in self.brokers:
                return False
            
            self.brokers[broker_id].reset_metrics()
            return True
    
    def reset_all_brokers(self) -> None:
        """Reset health metrics for all brokers"""
        with self._lock:
            for metrics in self.brokers.values():
                metrics.reset_metrics()
    
    def rank_brokers_by_health(self, candidate_brokers: Optional[List[str]] = None) -> List[str]:
        """
        Rank brokers by health metrics (most healthy first)
        
        Args:
            candidate_brokers: Optional list of broker IDs to consider, or None for all
            
        Returns:
            List[str]: Broker IDs ranked by health (best first)
        """
        with self._lock:
            # Filter to candidate brokers if provided
            brokers_to_rank = {}
            for broker_id, metrics in self.brokers.items():
                if candidate_brokers is None or broker_id in candidate_brokers:
                    brokers_to_rank[broker_id] = metrics
            
            # Create scoring function that considers multiple factors
            def broker_health_score(broker_id: str) -> float:
                metrics = brokers_to_rank[broker_id]
                
                # Calculate a score where higher is better
                failure_rate = metrics.get_failure_rate()
                response_time = metrics.get_average_response_time()
                
                # Weight failure rate more heavily than response time
                # A score of 1.0 is perfect, 0.0 is terrible
                score = (1.0 - failure_rate) * 0.7 + (1.0 / (1.0 + response_time / 10.0)) * 0.3
                
                # Heavily penalize degraded brokers
                if metrics.is_degraded():
                    score *= 0.2
                
                # Penalize brokers with pattern failures
                if metrics.has_failure_pattern():
                    score *= 0.7
                
                return score
            
            # Rank brokers by score
            return sorted(brokers_to_rank.keys(), key=broker_health_score, reverse=True)
    
    def get_broker_recommendation(self, asset_type: Optional[str] = None,
                                 candidate_brokers: Optional[List[str]] = None) -> Optional[str]:
        """
        Get recommended broker based on health metrics
        
        Args:
            asset_type: Optional asset type for specialized recommendations
            candidate_brokers: Optional list of broker IDs to consider, or None for all
            
        Returns:
            Optional[str]: Recommended broker ID or None if no healthy brokers
        """
        ranked_brokers = self.rank_brokers_by_health(candidate_brokers)
        
        # Return the healthiest broker, or None if no brokers
        return ranked_brokers[0] if ranked_brokers else None
