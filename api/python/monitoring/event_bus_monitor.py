#!/usr/bin/env python
"""
Event Bus Monitoring System

This module provides real-time monitoring of the event bus system,
including:
- Event flow rates and volume
- Processing time statistics
- Event queue backlog monitoring
- Event type distribution analysis
- Deadlock and starvation detection
- Performance anomaly detection
"""

import logging
import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger(__name__)

@dataclass
class EventMetrics:
    """Metrics for a single event type"""
    count: int = 0
    processing_times: List[float] = field(default_factory=list)
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    last_event_time: Optional[datetime] = None
    throughput_per_second: float = 0.0
    error_count: int = 0
    handler_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def update_processing_stats(self):
        """Update derived statistics based on raw data"""
        if self.processing_times:
            self.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            self.max_processing_time = max(self.processing_times)
            self.min_processing_time = min(self.processing_times)


class EventBusMonitor:
    """
    Monitors the event bus system for performance and health metrics.
    
    Provides real-time metrics on event flow, processing times, and
    system health, with alerting capabilities for anomalies.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        persistence_manager: Optional[PersistenceManager] = None,
        alert_thresholds: Optional[Dict[str, Any]] = None,
        metrics_window_size: int = 1000
    ):
        """
        Initialize the event bus monitor.
        
        Args:
            event_bus: The event bus to monitor
            persistence_manager: Optional persistence manager for storing metrics
            alert_thresholds: Dictionary of alert thresholds
            metrics_window_size: Maximum number of metrics to keep in memory
        """
        self.event_bus = event_bus
        self.persistence = persistence_manager
        self.metrics_window_size = metrics_window_size
        
        # Set default alert thresholds
        self.alert_thresholds = {
            'processing_time_ms': 100,  # Alert if processing takes longer than 100ms
            'queue_size': 100,  # Alert if queue exceeds 100 events
            'events_per_second': 1000,  # Alert if processing more than 1000 events/sec
            'error_rate': 0.05,  # Alert if error rate exceeds 5%
            'idle_threshold_sec': 60,  # Alert if no events for 60 seconds (possible deadlock)
            'std_dev_multiplier': 3.0  # Alert if metric exceeds 3 standard deviations
        }
        
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # Metrics tracking
        self.event_metrics: Dict[str, EventMetrics] = defaultdict(EventMetrics)
        self.global_metrics = EventMetrics()
        self.start_time = datetime.now()
        self.is_running = False
        self.monitoring_thread = None
        
        # Time-series metrics storage
        self.time_series_metrics = {
            'timestamps': deque(maxlen=metrics_window_size),
            'event_counts': deque(maxlen=metrics_window_size),
            'avg_processing_times': deque(maxlen=metrics_window_size),
            'queue_sizes': deque(maxlen=metrics_window_size),
            'error_rates': deque(maxlen=metrics_window_size)
        }
        
        # Performance baseline tracking
        self.baseline_metrics = {}
        self.anomalies_detected = []
        
        # Event tracking for timing
        self.event_start_times: Dict[str, datetime] = {}
        self.last_metrics_update = datetime.now()
        
        # Monitor lock to prevent concurrent updates
        self.lock = threading.Lock()
        
        logger.info("Event Bus Monitor initialized")
    
    def start(self, update_interval_sec: float = 5.0):
        """
        Start monitoring the event bus.
        
        Args:
            update_interval_sec: How frequently to update metrics, in seconds
        """
        if self.is_running:
            logger.warning("Event Bus Monitor is already running")
            return
        
        self.is_running = True
        
        # Subscribe to all events
        self.event_bus.subscribe_all(self._event_start_handler)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval_sec,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Event Bus Monitor started with {update_interval_sec}s update interval")
    
    def stop(self):
        """Stop monitoring the event bus"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe_all(self._event_start_handler)
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Event Bus Monitor stopped")
    
    def _event_start_handler(self, event: Event):
        """Record the start time of an event for timing"""
        # Record the receipt time
        now = datetime.now()
        self.event_start_times[event.event_id] = now
        
        # Subscribe to the completion of this specific event
        self.event_bus.subscribe(
            f"{event.event_id}_processed",
            lambda e: self._event_complete_handler(event, e)
        )
        
        # Set a timeout to check for hanging events
        threading.Timer(
            self.alert_thresholds['processing_time_ms'] / 1000 * 2,
            lambda: self._check_hanging_event(event.event_id)
        ).start()
    
    def _event_complete_handler(self, original_event: Event, completion_event: Event):
        """Record metrics for a completed event"""
        with self.lock:
            # Calculate processing time
            start_time = self.event_start_times.get(original_event.event_id)
            if not start_time:
                return  # Can't calculate without start time
            
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update metrics for this event type
            event_type = original_event.event_type
            event_metrics = self.event_metrics[event_type]
            event_metrics.count += 1
            event_metrics.processing_times.append(processing_time_ms)
            event_metrics.last_event_time = end_time
            
            # Keep only the most recent processing times
            if len(event_metrics.processing_times) > 100:
                event_metrics.processing_times = event_metrics.processing_times[-100:]
            
            # Update the statistics
            event_metrics.update_processing_stats()
            
            # Check if this was an error
            was_error = (
                original_event.event_type == EventType.ERROR_OCCURRED or
                (hasattr(completion_event, 'data') and 
                completion_event.data.get('status') == 'error')
            )
            
            if was_error:
                event_metrics.error_count += 1
            
            # Update global metrics
            self.global_metrics.count += 1
            self.global_metrics.processing_times.append(processing_time_ms)
            
            # Keep only the most recent processing times
            if len(self.global_metrics.processing_times) > 1000:
                self.global_metrics.processing_times = self.global_metrics.processing_times[-1000:]
            
            self.global_metrics.update_processing_stats()
            
            if was_error:
                self.global_metrics.error_count += 1
            
            # Remove event from tracking
            self.event_start_times.pop(original_event.event_id, None)
            
            # Unsubscribe from completion event
            self.event_bus.unsubscribe(
                f"{original_event.event_id}_processed",
                lambda e: self._event_complete_handler(original_event, e)
            )
            
            # Check for alerting conditions
            self._check_alert_conditions(
                event_type, processing_time_ms, event_metrics.error_count
            )
    
    def _check_hanging_event(self, event_id: str):
        """Check if an event is still being processed after timeout"""
        if event_id in self.event_start_times:
            # Event is still in processing, it might be hanging
            start_time = self.event_start_times[event_id]
            hang_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.warning(
                f"Potential hanging event detected (ID: {event_id}), "
                f"processing for {hang_time:.2f}ms"
            )
            
            # Don't remove it yet, just alert; the event might still complete
            
            # Record the anomaly
            self.anomalies_detected.append({
                'timestamp': datetime.now(),
                'type': 'hanging_event',
                'event_id': event_id,
                'hang_time_ms': hang_time
            })
    
    def _monitoring_loop(self, update_interval_sec: float):
        """Background thread for periodic metric updates"""
        while self.is_running:
            try:
                self._update_time_series_metrics()
                self._detect_anomalies()
                self._persist_metrics()
                
                # Sleep until next update
                time.sleep(update_interval_sec)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    def _update_time_series_metrics(self):
        """Update time-series metrics for trend analysis"""
        with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_metrics_update).total_seconds()
            
            if elapsed == 0:
                return  # Avoid division by zero
            
            # Calculate current event rate
            events_since_last = self.global_metrics.count - (
                0 if not self.time_series_metrics['event_counts'] 
                else self.time_series_metrics['event_counts'][-1]
            )
            current_rate = events_since_last / elapsed
            
            # Record time-series data
            self.time_series_metrics['timestamps'].append(now)
            self.time_series_metrics['event_counts'].append(self.global_metrics.count)
            self.time_series_metrics['avg_processing_times'].append(
                self.global_metrics.avg_processing_time
            )
            self.time_series_metrics['queue_sizes'].append(
                len(self.event_start_times)  # Approximate queue size
            )
            
            # Calculate and record error rate
            if self.global_metrics.count > 0:
                error_rate = self.global_metrics.error_count / self.global_metrics.count
            else:
                error_rate = 0
            self.time_series_metrics['error_rates'].append(error_rate)
            
            # Update event type throughput rates
            for event_type, metrics in self.event_metrics.items():
                if metrics.last_event_time and elapsed > 0:
                    metrics.throughput_per_second = metrics.count / elapsed
            
            # Update global throughput
            self.global_metrics.throughput_per_second = current_rate
            
            # Update timestamp
            self.last_metrics_update = now
    
    def _detect_anomalies(self):
        """Detect anomalies in metrics using statistical methods"""
        # Need enough data for meaningful statistical analysis
        if len(self.time_series_metrics['avg_processing_times']) < 10:
            return
        
        with self.lock:
            # Calculate baselines if not already set
            if not self.baseline_metrics:
                self._calculate_baseline_metrics()
            
            # Check for processing time anomalies
            times = list(self.time_series_metrics['avg_processing_times'])
            avg = sum(times) / len(times)
            stdev = statistics.stdev(times) if len(times) > 1 else 0
            
            if stdev > 0:
                latest = times[-1]
                z_score = abs(latest - avg) / stdev
                
                # Check if outside expected range (e.g., 3 standard deviations)
                if z_score > self.alert_thresholds['std_dev_multiplier']:
                    logger.warning(
                        f"Processing time anomaly detected: {latest:.2f}ms "
                        f"(z-score: {z_score:.2f})"
                    )
                    
                    # Record the anomaly
                    self.anomalies_detected.append({
                        'timestamp': datetime.now(),
                        'type': 'processing_time',
                        'value': latest,
                        'z_score': z_score,
                        'baseline': avg,
                        'stdev': stdev
                    })
            
            # Check for event rate anomalies
            if len(self.time_series_metrics['event_counts']) > 1:
                # Calculate event rate changes
                counts = list(self.time_series_metrics['event_counts'])
                rates = []
                
                for i in range(1, len(counts)):
                    if (self.time_series_metrics['timestamps'][i] - 
                        self.time_series_metrics['timestamps'][i-1]).total_seconds() > 0:
                        rate = (counts[i] - counts[i-1]) / (
                            self.time_series_metrics['timestamps'][i] - 
                            self.time_series_metrics['timestamps'][i-1]
                        ).total_seconds()
                        rates.append(rate)
                
                if rates:
                    avg_rate = sum(rates) / len(rates)
                    stdev_rate = statistics.stdev(rates) if len(rates) > 1 else 0
                    
                    if stdev_rate > 0 and rates:
                        latest_rate = rates[-1]
                        z_score = abs(latest_rate - avg_rate) / stdev_rate
                        
                        # Check if outside expected range
                        if z_score > self.alert_thresholds['std_dev_multiplier']:
                            logger.warning(
                                f"Event rate anomaly detected: {latest_rate:.2f} events/sec "
                                f"(z-score: {z_score:.2f})"
                            )
                            
                            # Record the anomaly
                            self.anomalies_detected.append({
                                'timestamp': datetime.now(),
                                'type': 'event_rate',
                                'value': latest_rate,
                                'z_score': z_score,
                                'baseline': avg_rate,
                                'stdev': stdev_rate
                            })
            
            # Check for idle system (possible deadlock)
            now = datetime.now()
            idle_threshold = timedelta(seconds=self.alert_thresholds['idle_threshold_sec'])
            
            # Get most recent event time
            latest_event_time = max(
                (m.last_event_time for m in self.event_metrics.values() 
                 if m.last_event_time is not None),
                default=None
            )
            
            if (latest_event_time and 
                now - latest_event_time > idle_threshold and
                len(self.event_start_times) > 0):
                # System has events in queue but nothing completing
                logger.warning(
                    f"Possible event processing deadlock detected. "
                    f"No events completed for {(now - latest_event_time).total_seconds():.1f}s "
                    f"with {len(self.event_start_times)} events in queue"
                )
                
                # Record the anomaly
                self.anomalies_detected.append({
                    'timestamp': now,
                    'type': 'possible_deadlock',
                    'idle_time_sec': (now - latest_event_time).total_seconds(),
                    'queue_size': len(self.event_start_times)
                })
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics for anomaly detection"""
        # Need enough data for meaningful baselines
        if len(self.time_series_metrics['avg_processing_times']) < 10:
            return
        
        with self.lock:
            # Calculate processing time baseline
            times = list(self.time_series_metrics['avg_processing_times'])
            self.baseline_metrics['processing_time'] = {
                'mean': sum(times) / len(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0
            }
            
            # Calculate event rate baseline
            if len(self.time_series_metrics['event_counts']) > 1:
                # Calculate event rate changes
                counts = list(self.time_series_metrics['event_counts'])
                timestamps = list(self.time_series_metrics['timestamps'])
                rates = []
                
                for i in range(1, len(counts)):
                    elapsed = (timestamps[i] - timestamps[i-1]).total_seconds()
                    if elapsed > 0:
                        rate = (counts[i] - counts[i-1]) / elapsed
                        rates.append(rate)
                
                if rates:
                    self.baseline_metrics['event_rate'] = {
                        'mean': sum(rates) / len(rates),
                        'stdev': statistics.stdev(rates) if len(rates) > 1 else 0
                    }
            
            # Calculate error rate baseline
            error_rates = list(self.time_series_metrics['error_rates'])
            self.baseline_metrics['error_rate'] = {
                'mean': sum(error_rates) / len(error_rates),
                'stdev': statistics.stdev(error_rates) if len(error_rates) > 1 else 0
            }
            
            logger.info("Calculated baseline metrics for anomaly detection")
    
    def _check_alert_conditions(self, 
                              event_type: str, 
                              processing_time_ms: float,
                              error_count: int):
        """Check if any alert conditions are met"""
        # Check processing time threshold
        if processing_time_ms > self.alert_thresholds['processing_time_ms']:
            logger.warning(
                f"Slow event processing for {event_type}: {processing_time_ms:.2f}ms "
                f"(threshold: {self.alert_thresholds['processing_time_ms']}ms)"
            )
        
        # Check queue size threshold
        queue_size = len(self.event_start_times)
        if queue_size > self.alert_thresholds['queue_size']:
            logger.warning(
                f"Large event queue detected: {queue_size} events "
                f"(threshold: {self.alert_thresholds['queue_size']})"
            )
        
        # Check event rate threshold
        if self.global_metrics.throughput_per_second > self.alert_thresholds['events_per_second']:
            logger.warning(
                f"High event rate detected: {self.global_metrics.throughput_per_second:.2f} events/sec "
                f"(threshold: {self.alert_thresholds['events_per_second']} events/sec)"
            )
        
        # Check error rate threshold
        if self.global_metrics.count > 0:
            error_rate = self.global_metrics.error_count / self.global_metrics.count
            if error_rate > self.alert_thresholds['error_rate']:
                logger.warning(
                    f"High error rate detected: {error_rate:.2%} "
                    f"(threshold: {self.alert_thresholds['error_rate']:.2%})"
                )
    
    def _persist_metrics(self):
        """Persist metrics to storage if persistence manager is available"""
        if not self.persistence:
            return
        
        try:
            # Create metrics snapshot
            now = datetime.now()
            
            metrics_data = {
                'timestamp': now.isoformat(),
                'uptime_seconds': (now - self.start_time).total_seconds(),
                'global': {
                    'total_events': self.global_metrics.count,
                    'avg_processing_time_ms': self.global_metrics.avg_processing_time,
                    'max_processing_time_ms': self.global_metrics.max_processing_time,
                    'min_processing_time_ms': self.global_metrics.min_processing_time,
                    'throughput_per_second': self.global_metrics.throughput_per_second,
                    'error_count': self.global_metrics.error_count,
                    'error_rate': (self.global_metrics.error_count / self.global_metrics.count 
                                 if self.global_metrics.count > 0 else 0)
                },
                'event_types': {},
                'queue_info': {
                    'current_size': len(self.event_start_times),
                    'oldest_event_age_ms': 0  # Will be updated below
                },
                'anomalies': self.anomalies_detected[-10:] if self.anomalies_detected else []
            }
            
            # Add event type specific metrics
            for event_type, metrics in self.event_metrics.items():
                metrics_data['event_types'][event_type] = {
                    'count': metrics.count,
                    'avg_processing_time_ms': metrics.avg_processing_time,
                    'max_processing_time_ms': metrics.max_processing_time,
                    'throughput_per_second': metrics.throughput_per_second,
                    'error_count': metrics.error_count,
                    'error_rate': (metrics.error_count / metrics.count 
                                 if metrics.count > 0 else 0)
                }
            
            # Calculate age of oldest event in queue
            if self.event_start_times:
                oldest_time = min(self.event_start_times.values())
                oldest_age_ms = (now - oldest_time).total_seconds() * 1000
                metrics_data['queue_info']['oldest_event_age_ms'] = oldest_age_ms
            
            # Store in persistence system
            self.persistence.insert_document('event_bus_metrics', metrics_data)
            
        except Exception as e:
            logger.error(f"Error persisting metrics: {str(e)}")
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of current metrics.
        
        Returns:
            Dictionary containing current metrics
        """
        with self.lock:
            now = datetime.now()
            
            snapshot = {
                'timestamp': now.isoformat(),
                'uptime_seconds': (now - self.start_time).total_seconds(),
                'global_metrics': {
                    'total_events': self.global_metrics.count,
                    'avg_processing_time_ms': self.global_metrics.avg_processing_time,
                    'max_processing_time_ms': self.global_metrics.max_processing_time,
                    'min_processing_time_ms': self.global_metrics.min_processing_time,
                    'events_per_second': self.global_metrics.throughput_per_second,
                    'error_rate': (self.global_metrics.error_count / self.global_metrics.count 
                                if self.global_metrics.count > 0 else 0)
                },
                'queue_size': len(self.event_start_times),
                'event_type_metrics': {},
                'anomalies_detected': len(self.anomalies_detected),
                'recent_anomalies': self.anomalies_detected[-5:] if self.anomalies_detected else []
            }
            
            # Add top 10 event types by volume
            sorted_events = sorted(
                self.event_metrics.items(),
                key=lambda x: x[1].count,
                reverse=True
            )[:10]
            
            for event_type, metrics in sorted_events:
                snapshot['event_type_metrics'][event_type] = {
                    'count': metrics.count,
                    'avg_processing_time_ms': metrics.avg_processing_time,
                    'events_per_second': metrics.throughput_per_second,
                    'error_rate': (metrics.error_count / metrics.count 
                                if metrics.count > 0 else 0)
                }
            
            return snapshot
    
    def get_event_type_metrics(self, event_type: str) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific event type.
        
        Args:
            event_type: The event type to get metrics for
            
        Returns:
            Dictionary containing metrics for the event type
        """
        with self.lock:
            if event_type not in self.event_metrics:
                return {
                    'count': 0,
                    'avg_processing_time_ms': 0,
                    'max_processing_time_ms': 0,
                    'min_processing_time_ms': 0,
                    'throughput_per_second': 0,
                    'error_count': 0,
                    'error_rate': 0
                }
            
            metrics = self.event_metrics[event_type]
            
            return {
                'count': metrics.count,
                'avg_processing_time_ms': metrics.avg_processing_time,
                'max_processing_time_ms': metrics.max_processing_time,
                'min_processing_time_ms': metrics.min_processing_time,
                'throughput_per_second': metrics.throughput_per_second,
                'error_count': metrics.error_count,
                'error_rate': (metrics.error_count / metrics.count 
                             if metrics.count > 0 else 0)
            }
    
    def get_time_series_data(self) -> Dict[str, List]:
        """
        Get time-series data for metrics visualization.
        
        Returns:
            Dictionary containing time-series data
        """
        with self.lock:
            # Convert deque objects to lists
            return {
                'timestamps': [t.isoformat() for t in self.time_series_metrics['timestamps']],
                'event_counts': list(self.time_series_metrics['event_counts']),
                'avg_processing_times': list(self.time_series_metrics['avg_processing_times']),
                'queue_sizes': list(self.time_series_metrics['queue_sizes']),
                'error_rates': list(self.time_series_metrics['error_rates'])
            }
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get list of detected anomalies.
        
        Returns:
            List of anomaly dictionaries
        """
        with self.lock:
            return self.anomalies_detected


# Global monitor instance
_global_monitor = None

def get_global_event_bus_monitor() -> EventBusMonitor:
    """
    Get the global event bus monitor.
    
    Returns:
        The global event bus monitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        from trading_bot.core.event_bus import get_global_event_bus
        from trading_bot.data.persistence import get_global_persistence_manager
        
        # Get dependencies
        event_bus = get_global_event_bus()
        try:
            persistence = get_global_persistence_manager()
        except:
            persistence = None
            logger.warning("Could not get global persistence manager, metrics will not be persisted")
        
        # Create monitor
        _global_monitor = EventBusMonitor(
            event_bus=event_bus,
            persistence_manager=persistence
        )
        
        # Start monitoring
        _global_monitor.start()
        
    return _global_monitor
