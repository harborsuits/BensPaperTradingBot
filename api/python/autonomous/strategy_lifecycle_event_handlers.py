#!/usr/bin/env python3
"""
Strategy Lifecycle Event Handlers

This module implements the event handlers for the Strategy Lifecycle Manager,
providing integration with the event-driven architecture of the trading system.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import from lifecycle components
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_strategy_lifecycle_manager, StrategyVersion, VersionStatus, VersionSource
)
from trading_bot.autonomous.strategy_lifecycle_extensions import get_strategy_lifecycle_extension
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class StrategyLifecycleEventTracker:
    """
    Tracks and manages events related to strategy lifecycle.
    Serves as a central integration point between the lifecycle manager and event system.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the event tracker.
        
        Args:
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus or EventBus()
        
        # Get lifecycle components
        self.lifecycle_manager = get_strategy_lifecycle_manager(event_bus)
        self.lifecycle_extension = get_strategy_lifecycle_extension(event_bus)
        
        # Event history
        self.events: Dict[str, List[Dict[str, Any]]] = {
            "version_created": [],
            "version_promoted": [],
            "version_deployed": [],
            "version_replaced": [],
            "version_rollback": [],
            "version_retired": []
        }
        
        # Maximum events to store in history
        self.max_events_per_type = 100
        
        # Register for events
        self._register_for_events()
        
        logger.info("Strategy Lifecycle Event Tracker initialized")
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        # Strategy optimization events
        self.event_bus.register(EventType.STRATEGY_OPTIMISED, self._handle_strategy_optimised)
        
        # Deployment events
        self.event_bus.register(EventType.STRATEGY_DEPLOYED_WITH_RISK, self._handle_strategy_deployed)
        self.event_bus.register(EventType.DEPLOYMENT_STATUS_CHANGED, self._handle_deployment_status_changed)
        
        # Performance events
        self.event_bus.register(EventType.PERFORMANCE_REPORT, self._handle_performance_report)
        
        # Candidate events
        self.event_bus.register("STRATEGY_VERSION_PROMOTED", self._handle_version_promoted)
        self.event_bus.register("STRATEGY_VERSION_REPLACED", self._handle_version_replaced)
        self.event_bus.register("STRATEGY_VERSION_ROLLBACK", self._handle_version_rollback)
        
        logger.info("Registered for events")
    
    def _handle_strategy_optimised(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle strategy optimised event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        strategy_id = data.get("strategy_id")
        
        if not strategy_id:
            return
        
        # Extract parameters and metrics
        parameters = data.get("parameters", {})
        metrics = data.get("metrics", {})
        parent_version = data.get("parent_version")
        
        # Log the event
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "source": "OPTIMIZATION",
            "parent_version": parent_version,
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "total_return_pct": metrics.get("total_return_pct", 0),
            "win_rate": metrics.get("win_rate", 0)
        }
        
        # Add to event history
        self._add_to_history("version_created", event_data)
    
    def _handle_strategy_deployed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle strategy deployed event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        strategy_id = data.get("strategy_id")
        deployment_id = data.get("deployment_id")
        
        if not strategy_id or not deployment_id:
            return
        
        # Extract metadata
        metadata = data.get("metadata", {})
        version_id = metadata.get("version_id")
        
        # If no version ID in metadata, this might be a legacy deployment
        if not version_id:
            # Try to get active version
            active_version = self.lifecycle_manager.get_active_version(strategy_id)
            
            if active_version:
                version_id = active_version.version_id
            else:
                # Create a new version for this deployment
                parameters = data.get("parameters", {})
                version_id = self.lifecycle_manager.track_strategy_version(
                    strategy_id=strategy_id,
                    parameters=parameters,
                    source=VersionSource.MANUAL,
                    metadata={"deployment_id": deployment_id}
                )
        
        # Update version status
        if version_id:
            self.lifecycle_manager.set_version_status(
                strategy_id=strategy_id,
                version_id=version_id,
                status=VersionStatus.DEPLOYED,
                reason=f"Deployed with ID {deployment_id}"
            )
        
        # Log the event
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "deployment_id": deployment_id,
            "version_id": version_id,
            "allocation": data.get("risk_params", {}).get("allocation_percentage", 0),
            "risk_level": data.get("risk_params", {}).get("risk_level", "MEDIUM")
        }
        
        # Add to event history
        self._add_to_history("version_deployed", event_data)
    
    def _handle_deployment_status_changed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle deployment status changed event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        strategy_id = data.get("strategy_id")
        deployment_id = data.get("deployment_id")
        new_status = data.get("new_status")
        reason = data.get("reason", "")
        
        if not strategy_id or not deployment_id or not new_status:
            return
        
        # Handle stopped or paused deployments
        if new_status in ["STOPPED", "PAUSED"]:
            # Get active version
            active_version = self.lifecycle_manager.get_active_version(strategy_id)
            
            if not active_version:
                return
            
            # Record deployment end
            active_version.record_deployment_end(f"Deployment {new_status.lower()}: {reason}")
            
            # If stopped, mark as retired
            if new_status == "STOPPED":
                self.lifecycle_manager.set_version_status(
                    strategy_id=strategy_id,
                    version_id=active_version.version_id,
                    status=VersionStatus.RETIRED,
                    reason=f"Deployment stopped: {reason}"
                )
                
                # Log the retirement event
                event_data = {
                    "timestamp": datetime.now().isoformat(),
                    "strategy_id": strategy_id,
                    "deployment_id": deployment_id,
                    "version_id": active_version.version_id,
                    "reason": reason
                }
                
                self._add_to_history("version_retired", event_data)
    
    def _handle_performance_report(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle performance report event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        strategy_id = data.get("strategy_id")
        
        if not strategy_id:
            return
        
        # Get metrics
        metrics = data.get("metrics", {})
        
        # Check if we should start a version succession check
        if metrics:
            # This is a lightweight check that just marks the strategy for evaluation
            # in the next monitoring cycle. The actual evaluation happens in the
            # lifecycle extension's monitoring loop.
            self.lifecycle_extension.last_succession_check[strategy_id] = datetime.min
    
    def _handle_version_promoted(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle version promoted event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Add to event history
        self._add_to_history("version_promoted", data)
    
    def _handle_version_replaced(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle version replaced event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Add to event history
        self._add_to_history("version_replaced", data)
    
    def _handle_version_rollback(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle version rollback event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Add to event history
        self._add_to_history("version_rollback", data)
    
    def _add_to_history(self, event_category: str, event_data: Dict[str, Any]) -> None:
        """
        Add event to history with limit enforcement.
        
        Args:
            event_category: Event category
            event_data: Event data
        """
        if event_category not in self.events:
            self.events[event_category] = []
        
        # Add to beginning of list (newest first)
        self.events[event_category].insert(0, event_data)
        
        # Trim if needed
        if len(self.events[event_category]) > self.max_events_per_type:
            self.events[event_category] = self.events[event_category][:self.max_events_per_type]
    
    def get_recent_events(self, 
                        event_category: Optional[str] = None,
                        strategy_id: Optional[str] = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events, optionally filtered.
        
        Args:
            event_category: Optional event category to filter by
            strategy_id: Optional strategy ID to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        results = []
        
        # If category specified, only get those events
        categories = [event_category] if event_category else self.events.keys()
        
        for category in categories:
            if category not in self.events:
                continue
                
            for event in self.events[category]:
                # Filter by strategy if requested
                if strategy_id and event.get("strategy_id") != strategy_id:
                    continue
                    
                # Add category to event data
                event_copy = event.copy()
                event_copy["event_category"] = category
                
                results.append(event_copy)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limit results
        return results[:limit]
    
    def get_strategy_lifecycle_summary(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get a summary of a strategy's lifecycle.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict with summary information
        """
        # Get versions
        versions = self.lifecycle_manager.get_strategy_versions(strategy_id)
        
        if not versions:
            return {"strategy_id": strategy_id, "versions": 0}
        
        # Get active version
        active_version = self.lifecycle_manager.get_active_version(strategy_id)
        active_id = active_version.version_id if active_version else None
        
        # Get version stats
        version_stats = {
            status.value: len(self.lifecycle_manager.get_versions_by_status(strategy_id, status))
            for status in VersionStatus
        }
        
        # Get deployment history
        deployments = self.get_recent_events("version_deployed", strategy_id)
        
        # Get key metrics from active version
        active_metrics = {}
        if active_version:
            active_metrics = active_version.get_key_metrics_summary()
        
        # Build summary
        summary = {
            "strategy_id": strategy_id,
            "versions": len(versions),
            "active_version": active_id,
            "version_stats": version_stats,
            "deployments": len(deployments),
            "first_version_date": None,
            "active_metrics": active_metrics
        }
        
        # Get first version date if available
        if versions:
            first_version = self.lifecycle_manager.get_version(strategy_id, versions[0])
            if first_version:
                summary["first_version_date"] = first_version.created_at.isoformat()
        
        return summary
    
    def get_promotion_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of versions that are promotion candidates.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of candidate versions
        """
        results = []
        
        # Check all strategies
        for strategy_id in self.lifecycle_manager.versions.keys():
            # Get candidate versions
            candidates = self.lifecycle_manager.get_versions_by_status(
                strategy_id, VersionStatus.CANDIDATE
            )
            
            for candidate in candidates:
                # Get key metrics
                metrics = candidate.get_key_metrics_summary()
                
                # Add to results
                results.append({
                    "strategy_id": strategy_id,
                    "version_id": candidate.version_id,
                    "created_at": candidate.created_at.isoformat(),
                    "source": candidate.source.value,
                    "metrics": metrics
                })
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Limit results
        return results[:limit]


# Singleton instance for global access
_strategy_lifecycle_event_tracker = None

def get_strategy_lifecycle_event_tracker(event_bus: Optional[EventBus] = None) -> StrategyLifecycleEventTracker:
    """
    Get singleton instance of strategy lifecycle event tracker.
    
    Args:
        event_bus: Event bus for communication
        
    Returns:
        StrategyLifecycleEventTracker instance
    """
    global _strategy_lifecycle_event_tracker
    
    if _strategy_lifecycle_event_tracker is None:
        _strategy_lifecycle_event_tracker = StrategyLifecycleEventTracker(event_bus)
        
    return _strategy_lifecycle_event_tracker


# Define custom event types for lifecycle events
def register_lifecycle_event_types(event_bus: EventBus) -> None:
    """
    Register lifecycle event types with event bus.
    
    Args:
        event_bus: Event bus to register with
    """
    # Define custom event types
    lifecycle_event_types = [
        "STRATEGY_VERSION_CREATED",
        "STRATEGY_VERSION_PROMOTED",
        "STRATEGY_VERSION_DEPLOYED",
        "STRATEGY_VERSION_REPLACED",
        "STRATEGY_VERSION_ROLLBACK",
        "STRATEGY_VERSION_RETIRED"
    ]
    
    # Register each type
    for event_type in lifecycle_event_types:
        if not hasattr(EventType, event_type):
            # Only add if not already defined
            setattr(EventType, event_type, event_type)
    
    logger.info("Registered lifecycle event types")
