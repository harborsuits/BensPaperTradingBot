#!/usr/bin/env python3
"""
Optimization History

This module provides functionality for tracking optimization history and
analyzing optimization effectiveness. It maintains a record of all optimization
attempts and their outcomes for each strategy.

Classes:
    OptimizationHistoryTracker: Tracks and analyzes optimization history
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import pandas as pd
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationEffectiveness(str, Enum):
    """Classification of optimization effectiveness."""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    MINIMAL_IMPROVEMENT = "minimal_improvement"
    NO_CHANGE = "no_change"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class OptimizationHistoryEntry:
    """
    Represents a single optimization history entry.
    """
    
    def __init__(
        self,
        strategy_id: str,
        old_version_id: str,
        new_version_id: str,
        timestamp: datetime,
        optimization_parameters: Dict[str, Any],
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        improvement: Dict[str, float],
        effectiveness: OptimizationEffectiveness,
        job_id: str
    ):
        """
        Initialize a new optimization history entry.
        
        Args:
            strategy_id: Strategy identifier
            old_version_id: Previous version ID
            new_version_id: New version ID after optimization
            timestamp: When optimization was performed
            optimization_parameters: Parameters used for optimization
            old_metrics: Performance metrics before optimization
            new_metrics: Performance metrics after optimization
            improvement: Calculated improvements for each metric
            effectiveness: Overall effectiveness classification
            job_id: ID of the optimization job
        """
        self.strategy_id = strategy_id
        self.old_version_id = old_version_id
        self.new_version_id = new_version_id
        self.timestamp = timestamp
        self.optimization_parameters = optimization_parameters
        self.old_metrics = old_metrics
        self.new_metrics = new_metrics
        self.improvement = improvement
        self.effectiveness = effectiveness
        self.job_id = job_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "old_version_id": self.old_version_id,
            "new_version_id": self.new_version_id,
            "timestamp": self.timestamp.isoformat(),
            "optimization_parameters": self.optimization_parameters,
            "old_metrics": self.old_metrics,
            "new_metrics": self.new_metrics,
            "improvement": self.improvement,
            "effectiveness": self.effectiveness.value,
            "job_id": self.job_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationHistoryEntry':
        """Create entry from dictionary representation."""
        return cls(
            strategy_id=data["strategy_id"],
            old_version_id=data["old_version_id"],
            new_version_id=data["new_version_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            optimization_parameters=data["optimization_parameters"],
            old_metrics=data["old_metrics"],
            new_metrics=data["new_metrics"],
            improvement=data["improvement"],
            effectiveness=OptimizationEffectiveness(data["effectiveness"]),
            job_id=data["job_id"]
        )


class OptimizationHistoryTracker:
    """
    Tracks and analyzes optimization history.
    
    This class maintains a record of all optimization attempts and their
    outcomes, providing insights into:
    - Which optimization methods work best for each strategy
    - How parameters affect performance
    - Overall effectiveness of the optimization process
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the history tracker.
        
        Args:
            storage_path: Path to store history data
        """
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".trading_bot", "optimization"
        )
        self.history_file = os.path.join(self.storage_path, "optimization_history.json")
        self.history: Dict[str, List[OptimizationHistoryEntry]] = {}
        self.lock = threading.RLock()
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing history
        self._load_history()
        
        # Performance metric importance (for calculating overall effectiveness)
        self.metric_weights = {
            "sharpe_ratio": 0.35,
            "sortino_ratio": 0.15,
            "win_rate": 0.15,
            "profit_factor": 0.10,
            "max_drawdown": 0.15,
            "volatility": 0.10
        }
        
        # Thresholds for effectiveness classification
        self.effectiveness_thresholds = {
            "significant_improvement": 0.15,  # 15% improvement
            "moderate_improvement": 0.05,     # 5% improvement
            "minimal_improvement": 0.01,      # 1% improvement
            "regression_threshold": -0.02     # 2% regression
        }
    
    def _load_history(self) -> None:
        """Load history from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                
                with self.lock:
                    for strategy_id, entries in history_data.items():
                        self.history[strategy_id] = [
                            OptimizationHistoryEntry.from_dict(entry)
                            for entry in entries
                        ]
                
                logger.info(f"Loaded optimization history for {len(self.history)} strategies")
            except Exception as e:
                logger.error(f"Error loading optimization history: {str(e)}")
                # Create backup of corrupted file
                if os.path.exists(self.history_file):
                    backup_file = f"{self.history_file}.bak.{int(time.time())}"
                    try:
                        os.rename(self.history_file, backup_file)
                        logger.info(f"Created backup of history file at {backup_file}")
                    except Exception as be:
                        logger.error(f"Error creating backup: {str(be)}")
    
    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            with self.lock:
                history_data = {
                    strategy_id: [entry.to_dict() for entry in entries]
                    for strategy_id, entries in self.history.items()
                }
            
            # Write to temporary file first, then rename for atomic update
            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, self.history_file)
            logger.debug(f"Saved optimization history for {len(self.history)} strategies")
        except Exception as e:
            logger.error(f"Error saving optimization history: {str(e)}")
    
    def add_entry(
        self,
        strategy_id: str,
        old_version_id: str,
        new_version_id: str,
        optimization_parameters: Dict[str, Any],
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        job_id: str
    ) -> OptimizationHistoryEntry:
        """
        Add a new optimization history entry.
        
        Args:
            strategy_id: Strategy identifier
            old_version_id: Previous version ID
            new_version_id: New version ID after optimization
            optimization_parameters: Parameters used for optimization
            old_metrics: Performance metrics before optimization
            new_metrics: Performance metrics after optimization
            job_id: ID of the optimization job
            
        Returns:
            The created history entry
        """
        # Calculate improvements
        improvement = self._calculate_improvement(old_metrics, new_metrics)
        
        # Determine effectiveness
        effectiveness = self._evaluate_effectiveness(improvement)
        
        # Create entry
        entry = OptimizationHistoryEntry(
            strategy_id=strategy_id,
            old_version_id=old_version_id,
            new_version_id=new_version_id,
            timestamp=datetime.now(),
            optimization_parameters=optimization_parameters,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            improvement=improvement,
            effectiveness=effectiveness,
            job_id=job_id
        )
        
        # Add to history
        with self.lock:
            if strategy_id not in self.history:
                self.history[strategy_id] = []
            
            self.history[strategy_id].append(entry)
        
        # Save to disk
        self._save_history()
        
        logger.info(
            f"Added optimization history entry for {strategy_id}: "
            f"{effectiveness.value} (job_id: {job_id})"
        )
        
        return entry
    
    def _calculate_improvement(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate improvement percentages for each metric.
        
        Args:
            old_metrics: Performance metrics before optimization
            new_metrics: Performance metrics after optimization
            
        Returns:
            Dictionary of improvement percentages
        """
        improvement = {}
        
        # Process each metric
        for metric, new_value in new_metrics.items():
            if metric not in old_metrics:
                continue
                
            old_value = old_metrics[metric]
            
            # Handle division by zero
            if old_value == 0:
                if new_value == 0:
                    improvement[metric] = 0.0
                elif new_value > 0:
                    improvement[metric] = 1.0  # 100% improvement
                else:
                    improvement[metric] = -1.0  # 100% regression
                continue
            
            # Calculate percentage change
            change = (new_value - old_value) / abs(old_value)
            
            # Special handling for metrics where lower is better
            if metric in ("max_drawdown", "volatility"):
                # For drawdown (negative number), improvement means less negative
                if old_value < 0 and new_value < 0:
                    # If both are negative, improvement means moving toward zero
                    change = (abs(old_value) - abs(new_value)) / abs(old_value)
                elif old_value > 0 and new_value > 0:
                    # If both are positive (unusual), improvement means decreasing
                    change = (old_value - new_value) / old_value
            
            improvement[metric] = change
        
        return improvement
    
    def _evaluate_effectiveness(self, improvement: Dict[str, float]) -> OptimizationEffectiveness:
        """
        Evaluate overall optimization effectiveness.
        
        Args:
            improvement: Dictionary of improvement percentages
            
        Returns:
            Effectiveness classification
        """
        if not improvement:
            return OptimizationEffectiveness.UNKNOWN
        
        # Calculate weighted improvement
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for metric, value in improvement.items():
            weight = self.metric_weights.get(metric, 0.1)
            weighted_sum += value * weight
            weight_sum += weight
        
        if weight_sum == 0:
            return OptimizationEffectiveness.UNKNOWN
            
        overall_improvement = weighted_sum / weight_sum
        
        # Classify effectiveness
        if overall_improvement >= self.effectiveness_thresholds["significant_improvement"]:
            return OptimizationEffectiveness.SIGNIFICANT_IMPROVEMENT
        elif overall_improvement >= self.effectiveness_thresholds["moderate_improvement"]:
            return OptimizationEffectiveness.MODERATE_IMPROVEMENT
        elif overall_improvement >= self.effectiveness_thresholds["minimal_improvement"]:
            return OptimizationEffectiveness.MINIMAL_IMPROVEMENT
        elif overall_improvement <= self.effectiveness_thresholds["regression_threshold"]:
            return OptimizationEffectiveness.REGRESSION
        else:
            return OptimizationEffectiveness.NO_CHANGE
    
    def get_strategy_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get optimization history for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of history entries as dictionaries
        """
        with self.lock:
            entries = self.history.get(strategy_id, [])
            return [entry.to_dict() for entry in entries]
    
    def get_all_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all optimization history.
        
        Returns:
            Dictionary mapping strategy IDs to lists of history entries
        """
        with self.lock:
            return {
                strategy_id: [entry.to_dict() for entry in entries]
                for strategy_id, entries in self.history.items()
            }
    
    def get_optimization_effectiveness_stats(self) -> Dict[str, Any]:
        """
        Get statistics on optimization effectiveness.
        
        Returns:
            Dictionary with effectiveness statistics
        """
        with self.lock:
            all_entries = [
                entry
                for entries in self.history.values()
                for entry in entries
            ]
        
        if not all_entries:
            return {
                "total_optimizations": 0,
                "effectiveness_counts": {},
                "effectiveness_percentages": {},
                "average_improvement": {}
            }
        
        # Count effectiveness categories
        effectiveness_counts = {
            effectiveness.value: 0
            for effectiveness in OptimizationEffectiveness
        }
        
        for entry in all_entries:
            effectiveness_counts[entry.effectiveness.value] += 1
        
        # Calculate percentages
        total = len(all_entries)
        effectiveness_percentages = {
            category: count / total
            for category, count in effectiveness_counts.items()
        }
        
        # Calculate average improvement by metric
        metrics = set()
        for entry in all_entries:
            metrics.update(entry.improvement.keys())
        
        average_improvement = {}
        for metric in metrics:
            values = [
                entry.improvement.get(metric, 0)
                for entry in all_entries
                if metric in entry.improvement
            ]
            
            if values:
                average_improvement[metric] = sum(values) / len(values)
            else:
                average_improvement[metric] = 0
        
        return {
            "total_optimizations": total,
            "effectiveness_counts": effectiveness_counts,
            "effectiveness_percentages": effectiveness_percentages,
            "average_improvement": average_improvement
        }
    
    def get_best_optimization_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Determine which optimization methods work best for each strategy.
        
        Returns:
            Dictionary mapping strategy types to most effective methods
        """
        # This will require integrating with the strategy metadata to get strategy types
        # For now, return a placeholder
        return {}
    
    def get_success_trend(self, window_size: int = 10) -> Dict[str, List[float]]:
        """
        Calculate trend in optimization success rates.
        
        Args:
            window_size: Number of optimizations in each window
            
        Returns:
            Dictionary with trend data
        """
        with self.lock:
            all_entries = [
                entry
                for entries in self.history.values()
                for entry in entries
            ]
        
        # Sort by timestamp
        all_entries.sort(key=lambda e: e.timestamp)
        
        if len(all_entries) < window_size:
            return {
                "timestamps": [],
                "success_rates": [],
                "average_improvements": []
            }
        
        # Calculate success rate in sliding windows
        timestamps = []
        success_rates = []
        average_improvements = []
        
        for i in range(0, len(all_entries) - window_size + 1):
            window = all_entries[i:i+window_size]
            
            # Count successful optimizations (at least minimal improvement)
            successful = sum(
                1 for e in window
                if e.effectiveness in (
                    OptimizationEffectiveness.MINIMAL_IMPROVEMENT,
                    OptimizationEffectiveness.MODERATE_IMPROVEMENT,
                    OptimizationEffectiveness.SIGNIFICANT_IMPROVEMENT
                )
            )
            
            success_rate = successful / window_size
            
            # Calculate average improvement for sharpe ratio if available
            sharpe_improvements = [
                e.improvement.get("sharpe_ratio", 0)
                for e in window
                if "sharpe_ratio" in e.improvement
            ]
            
            avg_improvement = (
                sum(sharpe_improvements) / len(sharpe_improvements)
                if sharpe_improvements else 0
            )
            
            # Use timestamp of last entry in window
            timestamps.append(window[-1].timestamp.isoformat())
            success_rates.append(success_rate)
            average_improvements.append(avg_improvement)
        
        return {
            "timestamps": timestamps,
            "success_rates": success_rates,
            "average_improvements": average_improvements
        }
    
    def analyze_strategy_improvement(self, strategy_id: str) -> Dict[str, Any]:
        """
        Analyze improvement trajectory for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Analysis results
        """
        with self.lock:
            entries = self.history.get(strategy_id, [])
        
        if not entries:
            return {
                "strategy_id": strategy_id,
                "total_optimizations": 0,
                "trajectory": "unknown",
                "metrics": {}
            }
        
        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)
        
        # Track metrics over time
        metrics_history = {}
        
        for entry in entries:
            for metric, value in entry.new_metrics.items():
                if metric not in metrics_history:
                    metrics_history[metric] = []
                
                metrics_history[metric].append((entry.timestamp, value))
        
        # Calculate improvement trajectories
        metric_trajectories = {}
        
        for metric, history in metrics_history.items():
            if len(history) < 2:
                continue
                
            # Calculate linear regression slope
            x = np.array([(ts - entries[0].timestamp).total_seconds() / 86400 for ts, _ in history])
            y = np.array([value for _, value in history])
            
            slope, _ = np.polyfit(x, y, 1)
            
            # Adjust slope for metrics where lower is better
            if metric in ("max_drawdown", "volatility"):
                slope = -slope
            
            metric_trajectories[metric] = {
                "slope": slope,
                "trend": "improving" if slope > 0 else "declining",
                "values": [value for _, value in history],
                "timestamps": [ts.isoformat() for ts, _ in history]
            }
        
        # Determine overall trajectory
        if not metric_trajectories:
            overall_trajectory = "unknown"
        else:
            # Weight by metric importance
            weighted_slopes = []
            
            for metric, trajectory in metric_trajectories.items():
                weight = self.metric_weights.get(metric, 0.1)
                weighted_slopes.append(trajectory["slope"] * weight)
            
            overall_slope = sum(weighted_slopes) / sum(self.metric_weights.get(m, 0.1) for m in metric_trajectories)
            
            if overall_slope > 0.01:
                overall_trajectory = "improving"
            elif overall_slope < -0.01:
                overall_trajectory = "declining"
            else:
                overall_trajectory = "stable"
        
        return {
            "strategy_id": strategy_id,
            "total_optimizations": len(entries),
            "trajectory": overall_trajectory,
            "metrics": metric_trajectories,
            "first_optimization": entries[0].timestamp.isoformat(),
            "last_optimization": entries[-1].timestamp.isoformat()
        }


# Singleton instance
_history_tracker = None


def get_optimization_history_tracker() -> OptimizationHistoryTracker:
    """
    Get the singleton instance of OptimizationHistoryTracker.
    
    Returns:
        OptimizationHistoryTracker instance
    """
    global _history_tracker
    
    if _history_tracker is None:
        _history_tracker = OptimizationHistoryTracker()
    
    return _history_tracker


if __name__ == "__main__":
    # Example usage
    tracker = get_optimization_history_tracker()
    
    # Add sample entry
    tracker.add_entry(
        strategy_id="iron_condor_spy",
        old_version_id="v1.0.0",
        new_version_id="v1.1.0",
        optimization_parameters={
            "method": "bayesian",
            "iterations": 100,
            "target_metric": "sharpe_ratio"
        },
        old_metrics={
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.12,
            "win_rate": 0.65
        },
        new_metrics={
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.72
        },
        job_id="sample_job_123"
    )
    
    # Get stats
    stats = tracker.get_optimization_effectiveness_stats()
    print(f"Optimization stats: {stats}")
