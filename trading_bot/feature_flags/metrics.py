#!/usr/bin/env python3
"""
Feature Flag Metrics Collector

Tracks performance metrics alongside feature flag state changes to measure
the impact of each feature on trading outcomes.
"""

import os
import json
import logging
import threading
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import pandas as pd
import numpy as np

# Optional Prometheus support
try:
    import prometheus_client as prom
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from .service import get_feature_flag_service, FlagChangeEvent, FeatureFlag

logger = logging.getLogger(__name__)


class FeatureFlagMetricsCollector:
    """Collects metrics about feature flag usage and performance."""
    
    _instance = None
    _instance_lock = threading.Lock()
    
    def __init__(
        self,
        metrics_dir: str = "data/feature_flags/metrics",
        performance_data_callback: Optional[Callable[[], Dict[str, float]]] = None,
        save_interval: int = 300,  # 5 minutes
        max_history: int = 10000
    ):
        """Initialize the metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics data
            performance_data_callback: Callback to get current performance metrics
            save_interval: How often to save metrics (seconds)
            max_history: Maximum number of data points to keep in memory
        """
        self.metrics_dir = metrics_dir
        self.performance_data_callback = performance_data_callback
        self.save_interval = save_interval
        self.max_history = max_history
        
        # Create metrics directory
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Initialize data structures
        self.flag_usage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.flag_changes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_snapshots: List[Dict[str, Any]] = []
        
        # Register for flag change events
        service = get_feature_flag_service()
        service.register_callback(self._on_flag_change)
        
        # Start background save thread
        self._save_stop_event = threading.Event()
        self._save_thread = threading.Thread(
            target=self._auto_save_worker,
            daemon=True,
            name="FeatureFlagMetricsSave"
        )
        self._save_thread.start()
        
        logger.info(f"Feature flag metrics collector initialized")
    
    def _auto_save_worker(self):
        """Background thread to periodically save metrics."""
        while not self._save_stop_event.wait(self.save_interval):
            try:
                self.save_metrics()
            except Exception as e:
                logger.error(f"Error saving metrics: {e}")
    
    def _on_flag_change(self, event: FlagChangeEvent):
        """Handle flag change events.
        
        Args:
            event: The flag change event
        """
        # Get current performance snapshot
        performance = self._get_performance_data()
        
        # Record the change with performance data
        change_data = {
            "timestamp": event.timestamp,
            "flag_id": event.flag_id,
            "enabled": event.enabled,
            "changed_by": event.changed_by,
            "reason": event.reason,
            "performance": performance
        }
        
        # Add to changes history
        self.flag_changes[event.flag_id].append(change_data)
        
        # Trim if necessary
        if len(self.flag_changes[event.flag_id]) > self.max_history:
            self.flag_changes[event.flag_id] = self.flag_changes[event.flag_id][-self.max_history:]
        
        logger.debug(f"Recorded metrics for flag change: {event.flag_id} -> {event.enabled}")
    
    def record_flag_usage(self, flag_id: str, context: Dict[str, Any], enabled: bool):
        """Record usage of a feature flag.
        
        Args:
            flag_id: ID of the flag
            context: The context in which the flag was checked
            enabled: Whether the flag was enabled
        """
        # Get current performance snapshot
        performance = self._get_performance_data()
        
        # Prepare usage data
        usage_data = {
            "timestamp": datetime.now(),
            "flag_id": flag_id,
            "enabled": enabled,
            "context": context,
            "performance": performance
        }
        
        # Add to usage history
        self.flag_usage[flag_id].append(usage_data)
        
        # Trim if necessary
        if len(self.flag_usage[flag_id]) > self.max_history:
            self.flag_usage[flag_id] = self.flag_usage[flag_id][-self.max_history:]
    
    def record_performance_snapshot(self):
        """Record a performance snapshot with current flag states."""
        # Get current flag states
        service = get_feature_flag_service()
        flag_states = {
            flag.id: service.is_enabled(flag.id)
            for flag in service.list_flags()
        }
        
        # Get performance data
        performance = self._get_performance_data()
        
        # Create snapshot
        snapshot = {
            "timestamp": datetime.now(),
            "flag_states": flag_states,
            "performance": performance
        }
        
        # Add to snapshots
        self.performance_snapshots.append(snapshot)
        
        # Trim if necessary
        if len(self.performance_snapshots) > self.max_history:
            self.performance_snapshots = self.performance_snapshots[-self.max_history:]
    
    def _get_performance_data(self) -> Dict[str, float]:
        """Get current performance data.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        if self.performance_data_callback:
            try:
                return self.performance_data_callback()
            except Exception as e:
                logger.error(f"Error getting performance data: {e}")
        
        # Return empty dict if no callback or error
        return {}
    
    def save_metrics(self):
        """Save all metrics to disk."""
        try:
            # Save flag usage data
            usage_file = os.path.join(self.metrics_dir, "flag_usage.json")
            with open(usage_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "data": {
                        flag_id: [
                            {**item, "timestamp": item["timestamp"].isoformat()} 
                            for item in data
                        ]
                        for flag_id, data in self.flag_usage.items()
                    }
                }, f, indent=2)
            
            # Save flag changes data
            changes_file = os.path.join(self.metrics_dir, "flag_changes.json")
            with open(changes_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "data": {
                        flag_id: [
                            {**item, "timestamp": item["timestamp"].isoformat()} 
                            for item in data
                        ]
                        for flag_id, data in self.flag_changes.items()
                    }
                }, f, indent=2)
            
            # Save performance snapshots
            snapshots_file = os.path.join(self.metrics_dir, "performance_snapshots.json")
            with open(snapshots_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "data": [
                        {**snapshot, "timestamp": snapshot["timestamp"].isoformat()}
                        for snapshot in self.performance_snapshots
                    ]
                }, f, indent=2)
            
            logger.debug(f"Saved feature flag metrics to {self.metrics_dir}")
        
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from disk."""
        try:
            # Load flag usage data
            usage_file = os.path.join(self.metrics_dir, "flag_usage.json")
            if os.path.exists(usage_file):
                with open(usage_file, 'r') as f:
                    data = json.load(f)
                    for flag_id, usage_data in data.get("data", {}).items():
                        self.flag_usage[flag_id] = [
                            {**item, "timestamp": datetime.fromisoformat(item["timestamp"])}
                            for item in usage_data
                        ]
            
            # Load flag changes data
            changes_file = os.path.join(self.metrics_dir, "flag_changes.json")
            if os.path.exists(changes_file):
                with open(changes_file, 'r') as f:
                    data = json.load(f)
                    for flag_id, changes_data in data.get("data", {}).items():
                        self.flag_changes[flag_id] = [
                            {**item, "timestamp": datetime.fromisoformat(item["timestamp"])}
                            for item in changes_data
                        ]
            
            # Load performance snapshots
            snapshots_file = os.path.join(self.metrics_dir, "performance_snapshots.json")
            if os.path.exists(snapshots_file):
                with open(snapshots_file, 'r') as f:
                    data = json.load(f)
                    self.performance_snapshots = [
                        {**snapshot, "timestamp": datetime.fromisoformat(snapshot["timestamp"])}
                        for snapshot in data.get("data", [])
                    ]
            
            logger.info(f"Loaded feature flag metrics from {self.metrics_dir}")
        
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def analyze_flag_impact(self, flag_id: str, metric_name: str, days: int = 30) -> Dict[str, Any]:
        """Analyze the impact of a feature flag on a specific metric.
        
        Args:
            flag_id: ID of the flag to analyze
            metric_name: Name of the metric to analyze
            days: Number of days of data to consider
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Convert snapshots to DataFrame
        snapshots = pd.DataFrame([
            {
                "timestamp": snapshot["timestamp"],
                "flag_enabled": snapshot["flag_states"].get(flag_id, False),
                "metric_value": snapshot["performance"].get(metric_name)
            }
            for snapshot in self.performance_snapshots
            if snapshot["timestamp"] >= datetime.now() - timedelta(days=days)
            and metric_name in snapshot["performance"]
        ])
        
        if snapshots.empty:
            return {
                "flag_id": flag_id,
                "metric": metric_name,
                "data_points": 0,
                "error": "No data available"
            }
        
        # Group by flag state
        grouped = snapshots.groupby("flag_enabled")
        
        # Calculate statistics
        stats = {}
        for state, group in grouped:
            stats[f"enabled_{state}"] = {
                "count": len(group),
                "mean": group["metric_value"].mean(),
                "median": group["metric_value"].median(),
                "std": group["metric_value"].std(),
                "min": group["metric_value"].min(),
                "max": group["metric_value"].max()
            }
        
        # Calculate impact
        if True in stats and False in stats:
            # Calculate percent difference
            enabled_mean = stats["enabled_True"]["mean"]
            disabled_mean = stats["enabled_False"]["mean"]
            
            if disabled_mean != 0:
                percent_diff = (enabled_mean - disabled_mean) / abs(disabled_mean) * 100
            else:
                percent_diff = np.nan
            
            # Run t-test if enough data points
            if stats["enabled_True"]["count"] > 5 and stats["enabled_False"]["count"] > 5:
                from scipy import stats as scipy_stats
                enabled_values = snapshots[snapshots["flag_enabled"]]["metric_value"]
                disabled_values = snapshots[~snapshots["flag_enabled"]]["metric_value"]
                
                t_stat, p_value = scipy_stats.ttest_ind(
                    enabled_values, 
                    disabled_values,
                    equal_var=False  # Welch's t-test
                )
                
                significant = p_value < 0.05
            else:
                t_stat = np.nan
                p_value = np.nan
                significant = False
            
            impact = {
                "absolute_diff": enabled_mean - disabled_mean,
                "percent_diff": percent_diff,
                "t_statistic": t_stat,
                "p_value": p_value,
                "statistically_significant": significant
            }
        else:
            impact = {
                "error": "Need both enabled and disabled data points for comparison"
            }
        
        return {
            "flag_id": flag_id,
            "metric": metric_name,
            "period_days": days,
            "total_data_points": len(snapshots),
            "statistics": stats,
            "impact": impact
        }
    
    def generate_impact_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a report of all flag impacts on all metrics.
        
        Args:
            days: Number of days of data to consider
            
        Returns:
            Dict[str, Any]: Report data
        """
        service = get_feature_flag_service()
        flags = service.list_flags()
        
        # Get unique metrics from snapshots
        metrics = set()
        for snapshot in self.performance_snapshots:
            metrics.update(snapshot["performance"].keys())
        
        # Generate report
        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "flags_analyzed": len(flags),
            "metrics_analyzed": len(metrics),
            "flag_impacts": {}
        }
        
        for flag in flags:
            flag_report = {
                "flag_name": flag.name,
                "flag_category": flag.category.name,
                "metrics": {}
            }
            
            for metric in metrics:
                try:
                    impact = self.analyze_flag_impact(flag.id, metric, days)
                    flag_report["metrics"][metric] = impact
                except Exception as e:
                    logger.error(f"Error analyzing impact of {flag.id} on {metric}: {e}")
                    flag_report["metrics"][metric] = {
                        "error": str(e)
                    }
            
            report["flag_impacts"][flag.id] = flag_report
        
        return report
    
    def cleanup(self):
        """Clean up resources used by the metrics collector."""
        # Stop background save thread
        if self._save_thread and self._save_thread.is_alive():
            self._save_stop_event.set()
            self._save_thread.join(timeout=5)
        
        # Save metrics one last time
        self.save_metrics()


def get_metrics_collector(
    performance_data_callback: Optional[Callable[[], Dict[str, float]]] = None
) -> FeatureFlagMetricsCollector:
    """Get the singleton metrics collector instance.
    
    Args:
        performance_data_callback: Callback to get current performance metrics
        
    Returns:
        FeatureFlagMetricsCollector: The metrics collector instance
    """
    with FeatureFlagMetricsCollector._instance_lock:
        if FeatureFlagMetricsCollector._instance is None:
            FeatureFlagMetricsCollector._instance = FeatureFlagMetricsCollector(
                performance_data_callback=performance_data_callback
            )
        elif performance_data_callback is not None:
            # Update the callback if provided
            FeatureFlagMetricsCollector._instance.performance_data_callback = performance_data_callback
            
        return FeatureFlagMetricsCollector._instance 