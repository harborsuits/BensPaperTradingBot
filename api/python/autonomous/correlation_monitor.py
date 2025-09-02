#!/usr/bin/env python3
"""
Correlation Monitor

This module monitors correlations between trading strategies and integrates with the
risk management system to adjust allocations based on correlation analysis.
"""

import os
import json
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import time

# Import correlation components
from trading_bot.autonomous.correlation_matrix import CorrelationMatrix

# Import risk and deployment components
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline, DeploymentStatus

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class CorrelationMonitor:
    """
    Monitors correlations between trading strategies and adjusts allocations
    based on correlation analysis to maintain diversification.
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 persistence_dir: Optional[str] = None):
        """
        Initialize the correlation monitor.
        
        Args:
            event_bus: Event bus for communication
            persistence_dir: Directory for persisting state
        """
        self.event_bus = event_bus or EventBus()
        
        # Directory for persisting state
        self.persistence_dir = persistence_dir or os.path.join(
            os.path.expanduser("~"), 
            ".trading_bot", 
            "correlation"
        )
        
        # Ensure directory exists
        os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Initialize correlation matrix
        self.correlation_matrix = CorrelationMatrix(
            window_size=30,
            min_periods=10,
            correlation_method='pearson'
        )
        
        # Get risk and deployment components
        self.risk_manager = get_autonomous_risk_manager()
        self.deployment_pipeline = get_deployment_pipeline()
        
        # Configuration
        self.config = {
            "high_correlation_threshold": 0.7,
            "allocation_reduction_factor": 0.8,
            "monitoring_interval_seconds": 3600,  # 1 hour
            "report_interval_hours": 24,          # 1 day
            "auto_adjust_allocations": True,
            "correlation_window_size": 30,        # 30 days
            "min_data_points": 10,                # At least 10 data points required
            "max_allocation_reduction": 0.5,      # Don't reduce below 50% of original
            "save_interval_hours": 6,             # Save state every 6 hours
            "regime_detection_enabled": False,    # Disable until implemented
            "regime_lookback_period": 90,         # 90 days for regime detection
        }
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.last_report_time = None
        self.last_save_time = None
        self._lock = threading.RLock()
        
        # Track correlation-based allocation adjustments
        self.allocation_adjustments: Dict[str, Dict[str, Any]] = {}
        
        # Event history (cached for UI)
        self.correlation_events: List[Dict[str, Any]] = []
        self.max_events = 100
        
        # Load state if available
        self._load_state()
        
        # Register for events
        self._register_for_events()
        
        logger.info("Correlation Monitor initialized")
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        # Register for trade execution events to collect return data
        self.event_bus.register(EventType.TRADE_EXECUTED, self._handle_trade_executed)
        self.event_bus.register(EventType.POSITION_CLOSED, self._handle_position_closed)
        
        # Register for strategy deployment events
        self.event_bus.register(EventType.STRATEGY_DEPLOYED_WITH_RISK, self._handle_strategy_deployed)
        self.event_bus.register(EventType.DEPLOYMENT_STATUS_CHANGED, self._handle_deployment_status_changed)
        
        # Register for performance report events
        self.event_bus.register(EventType.PERFORMANCE_REPORT, self._handle_performance_report)
        
        logger.info("Registered for events")
    
    def start_monitoring(self) -> None:
        """Start correlation monitoring."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started correlation monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop correlation monitoring."""
        if not self.is_running:
            logger.warning("Monitoring not running")
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped correlation monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Get current time
                now = datetime.now()
                
                # Calculate correlation matrix
                self.correlation_matrix.calculate_correlation()
                
                # Check for highly correlated pairs
                self._check_correlation_thresholds()
                
                # Adjust allocations if needed
                if self.config["auto_adjust_allocations"]:
                    self._adjust_allocations_for_correlation()
                
                # Generate correlation report if it's time
                if (not self.last_report_time or 
                    (now - self.last_report_time).total_seconds() / 3600 >= self.config["report_interval_hours"]):
                    self._generate_correlation_report()
                    self.last_report_time = now
                
                # Save state if needed
                if (not self.last_save_time or 
                    (now - self.last_save_time).total_seconds() / 3600 >= self.config["save_interval_hours"]):
                    self._save_state()
                    self.last_save_time = now
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep until next check
            time.sleep(self.config["monitoring_interval_seconds"])
    
    def _check_correlation_thresholds(self) -> None:
        """Check for strategy pairs exceeding correlation thresholds."""
        # Get highly correlated pairs
        threshold = self.config["high_correlation_threshold"]
        highly_correlated = self.correlation_matrix.get_highly_correlated_pairs(threshold)
        
        if not highly_correlated:
            return
        
        # Log and emit events for high correlations
        for strategy1, strategy2, correlation in highly_correlated:
            logger.warning(
                f"High correlation ({correlation:.2f}) detected between "
                f"{strategy1} and {strategy2}"
            )
            
            # Emit event
            self._emit_event(
                event_type="CORRELATION_THRESHOLD_EXCEEDED",
                data={
                    "strategy1": strategy1,
                    "strategy2": strategy2,
                    "correlation": correlation,
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _adjust_allocations_for_correlation(self) -> None:
        """Adjust strategy allocations based on correlation analysis."""
        # Get active deployments
        active_deployments = self.deployment_pipeline.get_deployments(
            status=DeploymentStatus.ACTIVE
        )
        
        if not active_deployments:
            return
        
        # Get current allocations
        current_allocations = {}
        for deployment in active_deployments:
            strategy_id = deployment.get("strategy_id")
            if not strategy_id:
                continue
                
            allocation = deployment.get("risk_params", {}).get("allocation_percentage", 5.0)
            current_allocations[strategy_id] = allocation
        
        # If fewer than 2 strategies, nothing to adjust
        if len(current_allocations) < 2:
            return
        
        # Calculate proposed adjustments
        adjustments = {}
        
        # Get highly correlated pairs
        threshold = self.config["high_correlation_threshold"]
        highly_correlated = self.correlation_matrix.get_highly_correlated_pairs(threshold)
        
        # Apply reductions for highly correlated pairs
        for strategy1, strategy2, correlation in highly_correlated:
            # Skip if either strategy is not active
            if strategy1 not in current_allocations or strategy2 not in current_allocations:
                continue
                
            # Determine which strategy to reduce
            # For simplicity, reduce the one with higher allocation
            if current_allocations[strategy1] >= current_allocations[strategy2]:
                reduce_strategy = strategy1
                other_strategy = strategy2
            else:
                reduce_strategy = strategy2
                other_strategy = strategy1
            
            # Calculate reduction factor based on correlation strength
            # Higher correlation = more reduction
            corr_factor = abs(correlation)
            reduction_factor = 1.0 - (corr_factor - threshold) / (1.0 - threshold) * (1.0 - self.config["allocation_reduction_factor"])
            
            # Don't reduce below minimum
            min_factor = self.config["max_allocation_reduction"]
            reduction_factor = max(min_factor, reduction_factor)
            
            # Record adjustment
            if reduce_strategy not in adjustments:
                adjustments[reduce_strategy] = {
                    "original": current_allocations[reduce_strategy],
                    "factor": reduction_factor,
                    "correlations": []
                }
            
            # Keep track of the lowest factor
            adjustments[reduce_strategy]["factor"] = min(
                adjustments[reduce_strategy]["factor"],
                reduction_factor
            )
            
            # Record correlation that led to adjustment
            adjustments[reduce_strategy]["correlations"].append({
                "other_strategy": other_strategy,
                "correlation": correlation
            })
        
        # Apply adjustments
        for strategy_id, adjustment in adjustments.items():
            # Calculate new allocation
            original = adjustment["original"]
            factor = adjustment["factor"]
            new_allocation = original * factor
            
            # Apply the adjustment
            if self.risk_manager.adjust_allocation(strategy_id, new_allocation):
                logger.info(
                    f"Adjusted allocation for {strategy_id} from {original:.1f}% to {new_allocation:.1f}% "
                    f"due to high correlation"
                )
                
                # Record the adjustment
                self.allocation_adjustments[strategy_id] = {
                    "timestamp": datetime.now().isoformat(),
                    "original_allocation": original,
                    "new_allocation": new_allocation,
                    "adjustment_factor": factor,
                    "correlations": adjustment["correlations"]
                }
                
                # Emit event
                self._emit_event(
                    event_type="ALLOCATION_ADJUSTED_FOR_CORRELATION",
                    data={
                        "strategy_id": strategy_id,
                        "original_allocation": original,
                        "new_allocation": new_allocation,
                        "adjustment_factor": factor,
                        "correlations": adjustment["correlations"],
                        "timestamp": datetime.now().isoformat()
                    }
                )
    
    def _generate_correlation_report(self) -> None:
        """Generate and emit correlation report."""
        # Get correlation statistics
        stats = self.correlation_matrix.get_correlation_stats()
        
        # Get highly correlated pairs
        threshold = self.config["high_correlation_threshold"]
        highly_correlated = self.correlation_matrix.get_highly_correlated_pairs(threshold)
        
        # Create report data
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "highly_correlated_pairs": [
                {
                    "strategy1": s1,
                    "strategy2": s2,
                    "correlation": c
                }
                for s1, s2, c in highly_correlated
            ],
            "allocation_adjustments": self.allocation_adjustments
        }
        
        # Emit report event
        self._emit_event(
            event_type="CORRELATION_REPORT_GENERATED",
            data=report
        )
        
        logger.info(f"Generated correlation report with {len(highly_correlated)} highly correlated pairs")
    
    def _handle_trade_executed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle trade execution event to track performance.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Extract relevant data
        strategy_id = data.get("strategy_id")
        timestamp = data.get("timestamp")
        pnl = data.get("profit_loss", 0.0)
        
        if not strategy_id or not timestamp:
            return
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # We use profit/loss as a proxy for daily return
        # In a real system, you'd calculate actual returns
        self._update_strategy_return(strategy_id, timestamp, pnl)
    
    def _handle_position_closed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle position closed event to track performance.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Extract relevant data
        strategy_id = data.get("strategy_id")
        timestamp = data.get("timestamp")
        pnl = data.get("profit_loss", 0.0)
        pnl_pct = data.get("profit_loss_pct", 0.0)
        
        if not strategy_id or not timestamp:
            return
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Use percentage return if available, otherwise raw P&L
        return_value = pnl_pct if pnl_pct != 0.0 else pnl
        
        # Update return data
        self._update_strategy_return(strategy_id, timestamp, return_value)
    
    def _handle_strategy_deployed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle strategy deployed event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # When a new strategy is deployed, we want to track it for correlation
        strategy_id = data.get("strategy_id")
        
        if not strategy_id:
            return
        
        logger.info(f"Started tracking correlation for newly deployed strategy {strategy_id}")
    
    def _handle_deployment_status_changed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle deployment status changed event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Clear allocation adjustments when a strategy is stopped
        strategy_id = data.get("strategy_id")
        new_status = data.get("new_status")
        
        if not strategy_id or not new_status:
            return
        
        if new_status == "STOPPED" and strategy_id in self.allocation_adjustments:
            self.allocation_adjustments.pop(strategy_id)
            logger.info(f"Cleared allocation adjustments for stopped strategy {strategy_id}")
    
    def _handle_performance_report(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle performance report event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Extract relevant data
        strategy_id = data.get("strategy_id")
        metrics = data.get("metrics", {})
        timestamp = data.get("timestamp")
        
        if not strategy_id or not metrics:
            return
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                timestamp = datetime.now()
        elif not timestamp:
            timestamp = datetime.now()
        
        # Extract daily return if available
        daily_return = metrics.get("daily_return_pct")
        
        if daily_return is not None:
            self._update_strategy_return(strategy_id, timestamp, daily_return)
    
    def _update_strategy_return(self, 
                               strategy_id: str, 
                               timestamp: datetime, 
                               return_value: float) -> None:
        """
        Update return data for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            timestamp: Timestamp for the return
            return_value: Return value
        """
        # Add to correlation matrix
        with self._lock:
            self.correlation_matrix.add_return_data(
                strategy_id=strategy_id,
                date=timestamp,
                return_value=return_value
            )
    
    def get_correlation(self, 
                      strategy1: str, 
                      strategy2: str) -> Optional[float]:
        """
        Get correlation between two strategies.
        
        Args:
            strategy1: First strategy ID
            strategy2: Second strategy ID
            
        Returns:
            Correlation value or None if not available
        """
        return self.correlation_matrix.get_correlation_for_pair(strategy1, strategy2)
    
    def get_correlation_report(self) -> Dict[str, Any]:
        """
        Get the latest correlation report.
        
        Returns:
            Correlation report dictionary
        """
        # Get correlation statistics
        stats = self.correlation_matrix.get_correlation_stats()
        
        # Get highly correlated pairs
        threshold = self.config["high_correlation_threshold"]
        highly_correlated = self.correlation_matrix.get_highly_correlated_pairs(threshold)
        
        # Create report data
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "highly_correlated_pairs": [
                {
                    "strategy1": s1,
                    "strategy2": s2,
                    "correlation": c
                }
                for s1, s2, c in highly_correlated
            ],
            "allocation_adjustments": self.allocation_adjustments,
            "tracked_strategies": list(self.correlation_matrix.returns_data.columns)
        }
        
        return report
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current correlation matrix.
        
        Returns:
            Dict representation of correlation matrix
        """
        # Use latest correlation matrix
        if self.correlation_matrix.latest_correlation.empty:
            self.correlation_matrix.calculate_correlation()
        
        if self.correlation_matrix.latest_correlation.empty:
            return {}
        
        # Convert to dict
        result = {}
        for strategy1 in self.correlation_matrix.latest_correlation.index:
            result[strategy1] = {}
            for strategy2 in self.correlation_matrix.latest_correlation.columns:
                result[strategy1][strategy2] = float(
                    self.correlation_matrix.latest_correlation.loc[strategy1, strategy2]
                )
        
        return result
    
    def get_correlation_heatmap_data(self) -> Dict[str, Any]:
        """
        Get data for correlation heatmap visualization.
        
        Returns:
            Dict with heatmap data
        """
        # Use latest correlation matrix
        if self.correlation_matrix.latest_correlation.empty:
            self.correlation_matrix.calculate_correlation()
        
        if self.correlation_matrix.latest_correlation.empty:
            return {
                "strategies": [],
                "matrix": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Get strategies
        strategies = list(self.correlation_matrix.latest_correlation.index)
        
        # Convert to list of lists for heatmap
        matrix = []
        for i, strategy1 in enumerate(strategies):
            row = []
            for j, strategy2 in enumerate(strategies):
                row.append(float(
                    self.correlation_matrix.latest_correlation.loc[strategy1, strategy2]
                ))
            matrix.append(row)
        
        return {
            "strategies": strategies,
            "matrix": matrix,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_correlation_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent correlation-related events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of correlation events
        """
        return self.correlation_events[:limit]
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit event to event bus and store in history.
        
        Args:
            event_type: Event type
            data: Event data
        """
        if not self.event_bus:
            return
            
        try:
            # Ensure timestamp is present
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
            
            # Create event
            event = Event(
                event_type=event_type,
                source="CorrelationMonitor",
                data=data,
                timestamp=datetime.now()
            )
            
            # Publish event
            self.event_bus.publish(event)
            
            # Store in history
            event_record = {
                "event_type": event_type,
                "timestamp": data["timestamp"],
                "data": data
            }
            
            self.correlation_events.insert(0, event_record)
            
            # Trim history if needed
            if len(self.correlation_events) > self.max_events:
                self.correlation_events = self.correlation_events[:self.max_events]
                
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    def _get_state_file_path(self) -> str:
        """Get path to state file."""
        return os.path.join(self.persistence_dir, "correlation_state.json")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        with self._lock:
            try:
                # Create state dictionary
                state = {
                    "config": self.config,
                    "correlation_matrix": self.correlation_matrix.to_dict(),
                    "allocation_adjustments": self.allocation_adjustments,
                    "correlation_events": self.correlation_events[:20],  # Limit saved events
                    "timestamp": datetime.now().isoformat()
                }
                
                # Write to file
                with open(self._get_state_file_path(), 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.info("Saved correlation monitor state to disk")
                
            except Exception as e:
                logger.error(f"Error saving state: {e}")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self._get_state_file_path()
        
        if not os.path.exists(state_file):
            logger.info("No state file found, starting with empty state")
            return
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore configuration
            if "config" in state:
                saved_config = state["config"]
                # Only update keys that exist in both
                for key in self.config:
                    if key in saved_config:
                        self.config[key] = saved_config[key]
            
            # Restore correlation matrix
            if "correlation_matrix" in state:
                self.correlation_matrix = CorrelationMatrix.from_dict(state["correlation_matrix"])
            
            # Restore allocation adjustments
            if "allocation_adjustments" in state:
                self.allocation_adjustments = state["allocation_adjustments"]
            
            # Restore events
            if "correlation_events" in state:
                self.correlation_events = state["correlation_events"]
            
            logger.info("Loaded correlation monitor state from disk")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Continue with default state


# Singleton instance for global access
_correlation_monitor = None

def get_correlation_monitor(event_bus: Optional[EventBus] = None,
                           persistence_dir: Optional[str] = None) -> CorrelationMonitor:
    """
    Get singleton instance of correlation monitor.
    
    Args:
        event_bus: Event bus for communication
        persistence_dir: Directory for persisting state
        
    Returns:
        CorrelationMonitor instance
    """
    global _correlation_monitor
    
    if _correlation_monitor is None:
        _correlation_monitor = CorrelationMonitor(event_bus, persistence_dir)
        
    return _correlation_monitor


# Define custom event types for correlation monitoring
def register_correlation_event_types(event_bus: EventBus) -> None:
    """
    Register correlation-related event types with event bus.
    
    Args:
        event_bus: Event bus to register with
    """
    # Define custom event types
    correlation_event_types = [
        "CORRELATION_THRESHOLD_EXCEEDED",
        "ALLOCATION_ADJUSTED_FOR_CORRELATION",
        "CORRELATION_REPORT_GENERATED",
        "REGIME_CHANGE_DETECTED"
    ]
    
    # Register each type
    for event_type in correlation_event_types:
        if not hasattr(EventType, event_type):
            # Only add if not already defined
            setattr(EventType, event_type, event_type)
    
    logger.info("Registered correlation event types")
