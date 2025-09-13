"""
System Health Monitor for tracking resource usage during staging.
"""
import os
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    """
    Monitors system health metrics like memory usage, CPU utilization,
    latency, and error rates during the staging period.
    
    This helps identify potential issues like memory leaks or performance
    degradation before moving to live trading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system health monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.service_registry = ServiceRegistry.get_instance()
        self.service_registry.register_service("system_health_monitor", self)
        
        self.config = config or {}
        self.memory_alert_threshold_mb = self.config.get("memory_alert_threshold_mb",.500)
        self.cpu_alert_threshold_pct = self.config.get("cpu_alert_threshold_pct", 70)
        self.sampling_interval_seconds = self.config.get("sampling_interval_seconds", 60)
        
        # Metrics storage
        self.metrics_history = {
            "timestamp": [],
            "memory_usage_mb": [],
            "cpu_usage_pct": [],
            "process_memory_mb": [],
            "thread_count": [],
            "open_file_count": [],
            "event_processing_latency_ms": []
        }
        
        # Event processing latency tracking
        self.event_timestamps = {}
        self.event_latencies = []
        
        # Error tracking
        self.error_counts = {}
        self.total_operations = 0
        
        # Initialize thread
        self.running = False
        self.monitor_thread = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info("System health monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
        logger.info("System health monitoring stopped")
    
    def register_alert_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self.running:
            try:
                self._collect_metrics()
                
                # Check for issues
                self._check_for_memory_leak()
                self._check_for_high_cpu()
                
                # Wait for next sample
                time.sleep(self.sampling_interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.sampling_interval_seconds)
    
    def _collect_metrics(self) -> None:
        """Collect system and process metrics."""
        now = datetime.now()
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Process metrics
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        process_memory = memory_info.rss / (1024 * 1024)  # Convert to MB
        thread_count = process.num_threads()
        open_files = len(process.open_files())
        
        # Calculate average event processing latency
        avg_latency = 0
        if self.event_latencies:
            avg_latency = sum(self.event_latencies) / len(self.event_latencies)
            self.event_latencies = []  # Reset after calculating average
        
        # Store metrics
        self.metrics_history["timestamp"].append(now)
        self.metrics_history["memory_usage_mb"].append(process_memory)
        self.metrics_history["cpu_usage_pct"].append(cpu_usage)
        self.metrics_history["process_memory_mb"].append(process_memory)
        self.metrics_history["thread_count"].append(thread_count)
        self.metrics_history["open_file_count"].append(open_files)
        self.metrics_history["event_processing_latency_ms"].append(avg_latency)
        
        # Limit history length to avoid excessive memory usage
        max_history = 10000  # About 1 week of 1-minute samples
        if len(self.metrics_history["timestamp"]) > max_history:
            for key in self.metrics_history:
                self.metrics_history[key] = self.metrics_history[key][-max_history:]
    
    def _check_for_memory_leak(self) -> None:
        """Check for potential memory leaks."""
        if len(self.metrics_history["memory_usage_mb"]) < 30:
            return  # Need more data points
            
        # Get last 30 memory readings
        recent_memory = self.metrics_history["memory_usage_mb"][-30:]
        
        # Check if memory is consistently increasing
        is_increasing = all(y > x for x, y in zip(recent_memory[:-1], recent_memory[1:]))
        
        # Check absolute memory usage
        current_memory = recent_memory[-1]
        
        if is_increasing and current_memory > self.memory_alert_threshold_mb:
            message = f"Potential memory leak detected! Memory usage: {current_memory:.2f} MB"
            logger.warning(message)
            self._send_alert("memory_leak", {
                "memory_usage_mb": current_memory,
                "trend": "increasing"
            })
    
    def _check_for_high_cpu(self) -> None:
        """Check for high CPU usage."""
        if not self.metrics_history["cpu_usage_pct"]:
            return
            
        current_cpu = self.metrics_history["cpu_usage_pct"][-1]
        
        if current_cpu > self.cpu_alert_threshold_pct:
            message = f"High CPU usage detected: {current_cpu:.2f}%"
            logger.warning(message)
            self._send_alert("high_cpu", {
                "cpu_usage_pct": current_cpu
            })
    
    def _send_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Send an alert to all registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def track_event_latency(self, event_id: str, start_time: Optional[float] = None) -> None:
        """
        Track event processing latency. Call once when event starts, and again when it completes.
        
        Args:
            event_id: Unique identifier for the event
            start_time: If provided, records this as the start time, otherwise uses current time
        """
        current_time = time.time() * 1000  # Convert to ms
        
        if event_id in self.event_timestamps:
            # Event is ending, calculate latency
            start_time = self.event_timestamps.pop(event_id)
            latency = current_time - start_time
            self.event_latencies.append(latency)
        else:
            # Event is starting
            self.event_timestamps[event_id] = start_time or current_time
    
    def track_error(self, error_type: str) -> None:
        """
        Track an error occurrence.
        
        Args:
            error_type: Type of error
        """
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.total_operations += 1
    
    def track_operation(self) -> None:
        """Track a successful operation for error rate calculation."""
        self.total_operations += 1
    
    def get_error_rate(self) -> float:
        """Get the overall error rate."""
        if self.total_operations == 0:
            return 0.0
            
        total_errors = sum(self.error_counts.values())
        return total_errors / self.total_operations
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics history as a pandas DataFrame."""
        return pd.DataFrame(self.metrics_history)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics."""
        if not self.metrics_history["timestamp"]:
            return {}
            
        return {
            "timestamp": self.metrics_history["timestamp"][-1],
            "memory_usage_mb": self.metrics_history["memory_usage_mb"][-1],
            "cpu_usage_pct": self.metrics_history["cpu_usage_pct"][-1],
            "process_memory_mb": self.metrics_history["process_memory_mb"][-1],
            "thread_count": self.metrics_history["thread_count"][-1],
            "open_file_count": self.metrics_history["open_file_count"][-1],
            "event_processing_latency_ms": self.metrics_history["event_processing_latency_ms"][-1],
            "error_rate": self.get_error_rate()
        }
    
    def detect_memory_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trend to detect potential issues."""
        if len(self.metrics_history["memory_usage_mb"]) < 60:
            return {"status": "insufficient_data"}
            
        # Get memory data as numpy array
        memory_data = np.array(self.metrics_history["memory_usage_mb"])
        
        # Fit linear regression to detect trend
        x = np.arange(len(memory_data))
        coeffs = np.polyfit(x, memory_data, 1)
        slope = coeffs[0]
        
        # Calculate projected memory in 24 hours if trend continues
        hours_24 = 24 * 60 * 60 / self.sampling_interval_seconds
        projected_memory = memory_data[-1] + (slope * hours_24)
        
        return {
            "status": "analyzed",
            "slope_mb_per_sample": slope,
            "trend": "increasing" if slope > 0.01 else "stable" if abs(slope) <= 0.01 else "decreasing",
            "projected_memory_24h_mb": projected_memory,
            "current_memory_mb": memory_data[-1]
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        if len(self.metrics_history["timestamp"]) < 10:
            return {"status": "insufficient_data"}
            
        # Convert to DataFrame for easier analysis
        df = self.get_metrics_dataframe()
        
        # Basic stats
        memory_stats = {
            "min": df["memory_usage_mb"].min(),
            "max": df["memory_usage_mb"].max(),
            "mean": df["memory_usage_mb"].mean(),
            "current": df["memory_usage_mb"].iloc[-1],
            "trend": self.detect_memory_trend()
        }
        
        cpu_stats = {
            "min": df["cpu_usage_pct"].min(),
            "max": df["cpu_usage_pct"].max(),
            "mean": df["cpu_usage_pct"].mean(),
            "current": df["cpu_usage_pct"].iloc[-1]
        }
        
        latency_stats = {
            "min": df["event_processing_latency_ms"].min(),
            "max": df["event_processing_latency_ms"].max(),
            "mean": df["event_processing_latency_ms"].mean(),
            "current": df["event_processing_latency_ms"].iloc[-1]
        }
        
        # Error statistics
        error_stats = {
            "total_errors": sum(self.error_counts.values()),
            "error_rate": self.get_error_rate(),
            "error_breakdown": self.error_counts.copy()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_duration_hours": len(df) * self.sampling_interval_seconds / 3600,
            "memory_stats": memory_stats,
            "cpu_stats": cpu_stats,
            "latency_stats": latency_stats,
            "error_stats": error_stats,
            "thread_count": df["thread_count"].iloc[-1],
            "open_file_count": df["open_file_count"].iloc[-1]
        }
