#!/usr/bin/env python3
"""
Strategy Lifecycle Manager

This module manages the entire lifecycle of trading strategies autonomously:
- Tracks strategy versions and their performance history
- Promotes strategies to deployment candidates based on criteria
- Manages strategy succession (replacing poor performers)
- Coordinates version management and rollback procedures
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import copy

# Import from existing components
from trading_bot.event_system import EventBus, Event, EventType
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline, DeploymentStatus
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.strategies.components.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)

class VersionStatus(Enum):
    """Status of a strategy version"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    CANDIDATE = "candidate"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class VersionSource(Enum):
    """Source of a strategy version"""
    INITIAL = "initial"
    MANUAL = "manual"
    OPTIMIZATION = "optimization"
    AUTO_ADJUSTMENT = "auto_adjustment"
    ROLLBACK = "rollback"


class StrategyVersion:
    """
    Represents a specific version of a strategy with its parameters and performance metrics.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 version_id: str,
                 parameters: Dict[str, Any],
                 created_at: Optional[datetime] = None,
                 source: VersionSource = VersionSource.INITIAL,
                 parent_version: Optional[str] = None,
                 status: VersionStatus = VersionStatus.DEVELOPMENT,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a strategy version.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            parameters: Strategy parameters
            created_at: Creation timestamp
            source: Source of this version
            parent_version: Parent version ID if this was derived
            status: Current status of this version
            metadata: Additional metadata
        """
        self.strategy_id = strategy_id
        self.version_id = version_id
        self.parameters = parameters
        self.created_at = created_at or datetime.now()
        self.source = source
        self.parent_version = parent_version
        self.status = status
        self.metadata = metadata or {}
        
        # Performance metrics
        self.metrics = {
            "backtest": {},  # Backtest results
            "live": {},      # Live trading results
            "deployments": [],  # History of deployments
        }
        
        # Timestamps for status changes
        self.status_history = [{
            "status": status.value,
            "timestamp": self.created_at.isoformat(),
            "reason": "Initial creation"
        }]
        
        # Performance comparison to parent
        self.improvement = {}
        
        # Usage stats
        self.last_deployed = None
        self.total_deployed_time = timedelta(0)
        self.deployment_count = 0
    
    def update_status(self, new_status: VersionStatus, reason: str = "") -> None:
        """
        Update the status of this version.
        
        Args:
            new_status: New status
            reason: Reason for status change
        """
        # Record the status change
        self.status_history.append({
            "status": new_status.value,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })
        
        # Update the status
        self.status = new_status
        
        # If deployed, update deployment stats
        if new_status == VersionStatus.DEPLOYED:
            self.last_deployed = datetime.now()
            self.deployment_count += 1
            
            # Add to deployment history
            self.metrics["deployments"].append({
                "deployed_at": self.last_deployed.isoformat(),
                "reason": reason
            })
        
        logger.info(f"Strategy {self.strategy_id} version {self.version_id} status changed to {new_status.value}: {reason}")
    
    def update_backtest_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update backtest metrics for this version.
        
        Args:
            metrics: Backtest performance metrics
        """
        self.metrics["backtest"].update(metrics)
        
        # If this version has a parent, calculate improvement
        if self.parent_version:
            # This would normally be done by comparing with the parent's metrics
            # but that would require access to the parent version object
            pass
    
    def update_live_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update live trading metrics for this version.
        
        Args:
            metrics: Live performance metrics
        """
        # Store the timestamp
        metrics["updated_at"] = datetime.now().isoformat()
        
        # Update metrics
        self.metrics["live"].update(metrics)
    
    def record_deployment_end(self, reason: str = "") -> None:
        """
        Record the end of a deployment period.
        
        Args:
            reason: Reason for ending deployment
        """
        if self.last_deployed:
            # Calculate deployment duration
            now = datetime.now()
            duration = now - self.last_deployed
            self.total_deployed_time += duration
            
            # Update the latest deployment record
            if self.metrics["deployments"]:
                self.metrics["deployments"][-1].update({
                    "ended_at": now.isoformat(),
                    "duration_hours": duration.total_seconds() / 3600,
                    "end_reason": reason
                })
            
            self.last_deployed = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict representation of this version
        """
        return {
            "strategy_id": self.strategy_id,
            "version_id": self.version_id,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "source": self.source.value,
            "parent_version": self.parent_version,
            "status": self.status.value,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "status_history": self.status_history,
            "improvement": self.improvement,
            "last_deployed": self.last_deployed.isoformat() if self.last_deployed else None,
            "total_deployed_time": self.total_deployed_time.total_seconds(),
            "deployment_count": self.deployment_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyVersion':
        """
        Create from dictionary representation.
        
        Args:
            data: Dict representation
            
        Returns:
            StrategyVersion instance
        """
        # Create instance with basic data
        version = cls(
            strategy_id=data["strategy_id"],
            version_id=data["version_id"],
            parameters=data["parameters"],
            created_at=datetime.fromisoformat(data["created_at"]),
            source=VersionSource(data["source"]),
            parent_version=data["parent_version"],
            status=VersionStatus(data["status"]),
            metadata=data["metadata"]
        )
        
        # Restore additional data
        version.metrics = data["metrics"]
        version.status_history = data["status_history"]
        version.improvement = data["improvement"]
        version.last_deployed = datetime.fromisoformat(data["last_deployed"]) if data["last_deployed"] else None
        version.total_deployed_time = timedelta(seconds=data["total_deployed_time"])
        version.deployment_count = data["deployment_count"]
        
        return version
    
    def get_key_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key performance metrics.
        
        Returns:
            Dict with key metrics
        """
        # Extract metrics from backtest and live data
        summary = {
            "version_id": self.version_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "deployment_count": self.deployment_count
        }
        
        # Add backtest metrics if available
        if self.metrics["backtest"]:
            backtest = self.metrics["backtest"]
            summary.update({
                "backtest_sharpe": backtest.get("sharpe_ratio", 0),
                "backtest_return": backtest.get("total_return_pct", 0),
                "backtest_drawdown": backtest.get("max_drawdown_pct", 0),
                "backtest_win_rate": backtest.get("win_rate", 0)
            })
        
        # Add live metrics if available
        if self.metrics["live"]:
            live = self.metrics["live"]
            summary.update({
                "live_sharpe": live.get("sharpe_ratio", 0),
                "live_return": live.get("total_return_pct", 0),
                "live_drawdown": live.get("max_drawdown_pct", 0),
                "live_win_rate": live.get("win_rate", 0)
            })
        
        return summary


class StrategyLifecycleManager:
    """
    Manages the lifecycle of trading strategies, including:
    - Version tracking
    - Performance history
    - Promotion and succession
    - Version management
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 persistence_dir: Optional[str] = None):
        """
        Initialize the strategy lifecycle manager.
        
        Args:
            event_bus: Event bus for communication
            persistence_dir: Directory for persisting state
        """
        self.event_bus = event_bus or EventBus()
        
        # Directory for persisting state
        self.persistence_dir = persistence_dir or os.path.join(
            os.path.expanduser("~"), 
            ".trading_bot", 
            "strategy_lifecycle"
        )
        
        # Ensure directory exists
        os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Strategy versions by strategy_id and version_id
        self.versions: Dict[str, Dict[str, StrategyVersion]] = {}
        
        # Track active versions
        self.active_versions: Dict[str, str] = {}
        
        # Internal state
        self._lock = threading.RLock()
        self._save_timer = None
        self._state_modified = False
        
        # Load state from disk
        self._load_state()
        
        # Register for events
        self._register_for_events()
        
        logger.info("Strategy Lifecycle Manager initialized")
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        # Strategy optimization events
        self.event_bus.register(EventType.STRATEGY_OPTIMISED, self._handle_strategy_optimised)
        
        # Deployment events
        self.event_bus.register(EventType.STRATEGY_DEPLOYED_WITH_RISK, self._handle_strategy_deployed)
        self.event_bus.register(EventType.DEPLOYMENT_STATUS_CHANGED, self._handle_deployment_status_changed)
        
        # Performance events
        self.event_bus.register(EventType.PERFORMANCE_REPORT, self._handle_performance_report)
        
        logger.info("Registered for events")
    
    def _get_state_file_path(self) -> str:
        """Get path to state file."""
        return os.path.join(self.persistence_dir, "lifecycle_state.json")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        if not self._state_modified:
            return
            
        with self._lock:
            try:
                # Convert versions to dict representation
                serialized_versions = {}
                for strategy_id, versions in self.versions.items():
                    serialized_versions[strategy_id] = {
                        version_id: version.to_dict()
                        for version_id, version in versions.items()
                    }
                
                # Create state dictionary
                state = {
                    "versions": serialized_versions,
                    "active_versions": self.active_versions,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Write to file
                with open(self._get_state_file_path(), 'w') as f:
                    json.dump(state, f, indent=2)
                
                self._state_modified = False
                logger.info("Saved state to disk")
                
            except Exception as e:
                logger.error(f"Error saving state: {e}")
    
    def _schedule_save(self) -> None:
        """Schedule state save with debouncing."""
        self._state_modified = True
        
        # Cancel existing timer
        if self._save_timer:
            self._save_timer.cancel()
        
        # Schedule save after delay
        self._save_timer = threading.Timer(5.0, self._save_state)
        self._save_timer.daemon = True
        self._save_timer.start()
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self._get_state_file_path()
        
        if not os.path.exists(state_file):
            logger.info("No state file found, starting with empty state")
            return
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore versions
            serialized_versions = state.get("versions", {})
            self.versions = {}
            
            for strategy_id, versions in serialized_versions.items():
                self.versions[strategy_id] = {}
                for version_id, version_data in versions.items():
                    self.versions[strategy_id][version_id] = StrategyVersion.from_dict(version_data)
            
            # Restore active versions
            self.active_versions = state.get("active_versions", {})
            
            logger.info(f"Loaded state with {len(self.versions)} strategies and {sum(len(v) for v in self.versions.values())} versions")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Start with empty state
            self.versions = {}
            self.active_versions = {}
    
    def track_strategy_version(self, 
                              strategy_id: str, 
                              parameters: Dict[str, Any],
                              metrics: Optional[Dict[str, Any]] = None,
                              source: VersionSource = VersionSource.INITIAL,
                              parent_version: Optional[str] = None,
                              version_id: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Track a new version of a strategy.
        
        Args:
            strategy_id: Strategy identifier
            parameters: Strategy parameters
            metrics: Initial performance metrics
            source: Source of this version
            parent_version: Parent version ID if this was derived
            version_id: Optional explicit version ID
            metadata: Additional metadata
            
        Returns:
            Version ID of the created version
        """
        with self._lock:
            # Create version ID if not provided
            if not version_id:
                version_id = f"v{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
            
            # Make sure we have a dict for this strategy
            if strategy_id not in self.versions:
                self.versions[strategy_id] = {}
            
            # Create the version
            version = StrategyVersion(
                strategy_id=strategy_id,
                version_id=version_id,
                parameters=parameters,
                source=source,
                parent_version=parent_version,
                metadata=metadata
            )
            
            # Add initial metrics if provided
            if metrics:
                version.update_backtest_metrics(metrics)
            
            # Store the version
            self.versions[strategy_id][version_id] = version
            
            # Schedule state save
            self._schedule_save()
            
            logger.info(f"Tracked new version {version_id} for strategy {strategy_id}")
            
            return version_id
    
    def get_strategy_versions(self, strategy_id: str) -> List[str]:
        """
        Get all versions for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of version IDs
        """
        with self._lock:
            if strategy_id not in self.versions:
                return []
                
            return list(self.versions[strategy_id].keys())
    
    def get_version(self, strategy_id: str, version_id: str) -> Optional[StrategyVersion]:
        """
        Get a specific version of a strategy.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            
        Returns:
            StrategyVersion or None if not found
        """
        with self._lock:
            if strategy_id not in self.versions:
                return None
                
            return self.versions[strategy_id].get(version_id)
    
    def get_latest_version(self, strategy_id: str) -> Optional[StrategyVersion]:
        """
        Get the latest version of a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Latest StrategyVersion or None if not found
        """
        with self._lock:
            if strategy_id not in self.versions:
                return None
                
            # Get all versions
            versions = list(self.versions[strategy_id].values())
            
            # Sort by creation time
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            return versions[0] if versions else None
    
    def get_active_version(self, strategy_id: str) -> Optional[StrategyVersion]:
        """
        Get the currently active version of a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Active StrategyVersion or None if not found
        """
        with self._lock:
            # Check if we have an active version
            version_id = self.active_versions.get(strategy_id)
            
            if not version_id:
                return None
                
            # Get the version
            return self.get_version(strategy_id, version_id)
    
    def update_version_metrics(self, 
                              strategy_id: str, 
                              version_id: str, 
                              metrics: Dict[str, Any],
                              is_live: bool = False) -> bool:
        """
        Update metrics for a version.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            metrics: Performance metrics
            is_live: Whether these are live or backtest metrics
            
        Returns:
            True if successful
        """
        with self._lock:
            # Get the version
            version = self.get_version(strategy_id, version_id)
            
            if not version:
                logger.warning(f"Cannot update metrics for unknown version {version_id} of strategy {strategy_id}")
                return False
            
            # Update the metrics
            if is_live:
                version.update_live_metrics(metrics)
            else:
                version.update_backtest_metrics(metrics)
            
            # Schedule state save
            self._schedule_save()
            
            return True
    
    def set_version_status(self, 
                          strategy_id: str, 
                          version_id: str, 
                          status: VersionStatus,
                          reason: str = "") -> bool:
        """
        Set the status of a version.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            status: New status
            reason: Reason for status change
            
        Returns:
            True if successful
        """
        with self._lock:
            # Get the version
            version = self.get_version(strategy_id, version_id)
            
            if not version:
                logger.warning(f"Cannot set status for unknown version {version_id} of strategy {strategy_id}")
                return False
            
            # Update the status
            version.update_status(status, reason)
            
            # If this is now deployed, make it active
            if status == VersionStatus.DEPLOYED:
                self.active_versions[strategy_id] = version_id
            
            # If this was deployed and is now retired, update deployment end
            if status == VersionStatus.RETIRED and version.last_deployed:
                version.record_deployment_end(reason)
                
                # Remove from active versions if it's still there
                if self.active_versions.get(strategy_id) == version_id:
                    self.active_versions.pop(strategy_id)
            
            # Schedule state save
            self._schedule_save()
            
            return True
    
    def get_versions_by_status(self, strategy_id: str, status: VersionStatus) -> List[StrategyVersion]:
        """
        Get all versions with a specific status.
        
        Args:
            strategy_id: Strategy identifier
            status: Status to filter by
            
        Returns:
            List of matching StrategyVersion objects
        """
        with self._lock:
            if strategy_id not in self.versions:
                return []
                
            return [
                version for version in self.versions[strategy_id].values()
                if version.status == status
            ]
    
    def get_best_performing_version(self, 
                                   strategy_id: str, 
                                   metric_name: str = "sharpe_ratio",
                                   min_backtest_threshold: float = 0.0,
                                   exclude_deployed: bool = True) -> Optional[StrategyVersion]:
        """
        Get the best performing version based on a metric.
        
        Args:
            strategy_id: Strategy identifier
            metric_name: Metric to compare by
            min_backtest_threshold: Minimum threshold for the metric
            exclude_deployed: Whether to exclude currently deployed versions
            
        Returns:
            Best performing StrategyVersion or None
        """
        with self._lock:
            if strategy_id not in self.versions:
                return None
                
            # Get all versions
            versions = list(self.versions[strategy_id].values())
            
            # Filter by status if needed
            if exclude_deployed:
                versions = [v for v in versions if v.status != VersionStatus.DEPLOYED]
            
            # Filter by minimum threshold
            versions = [
                v for v in versions 
                if v.metrics["backtest"].get(metric_name, 0) >= min_backtest_threshold
            ]
            
            if not versions:
                return None
                
            # Sort by the metric
            versions.sort(
                key=lambda v: v.metrics["backtest"].get(metric_name, 0), 
                reverse=True
            )
            
            return versions[0]


# Singleton instance for global access
_strategy_lifecycle_manager = None

def get_strategy_lifecycle_manager(event_bus: Optional[EventBus] = None,
                                 persistence_dir: Optional[str] = None) -> StrategyLifecycleManager:
    """
    Get singleton instance of strategy lifecycle manager.
    
    Args:
        event_bus: Event bus for communication
        persistence_dir: Directory for persisting state
        
    Returns:
        StrategyLifecycleManager instance
    """
    global _strategy_lifecycle_manager
    
    if _strategy_lifecycle_manager is None:
        _strategy_lifecycle_manager = StrategyLifecycleManager(event_bus, persistence_dir)
        
    return _strategy_lifecycle_manager
