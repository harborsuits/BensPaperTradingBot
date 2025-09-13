"""
Automated Feature Flag Rollback System

This module provides a system for automatically rolling back feature flags
when performance metrics exceed defined thresholds. This helps prevent
significant losses from unstable or poorly performing features.
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union

from .service import get_feature_flag_service, FlagCategory
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)

class ThresholdDirection(Enum):
    """Direction for a metric threshold."""
    ABOVE = "above"       # Trigger when metric goes above threshold
    BELOW = "below"       # Trigger when metric goes below threshold


class AlertSeverity(Enum):
    """Severity level for metric alerts."""
    INFO = "info"         # Informational only
    WARNING = "warning"   # Warning level
    CRITICAL = "critical" # Critical level - requires immediate action


class RollbackRule:
    """Defines a rule for when to roll back a feature flag."""
    
    def __init__(
        self,
        id: str,
        metric_name: str,
        threshold: float,
        direction: ThresholdDirection,
        severity: AlertSeverity = AlertSeverity.WARNING,
        evaluation_window: int = 5,  # minutes
        cooldown_period: int = 60,   # minutes
        notification_channels: Optional[List[str]] = None,
        description: str = ""
    ):
        """Initialize a rollback rule.
        
        Args:
            id: Unique identifier for the rule
            metric_name: Name of the metric to monitor
            threshold: Threshold value that triggers the rule
            direction: Whether to trigger above or below the threshold
            severity: Severity level of alerts
            evaluation_window: Time window (minutes) to evaluate the metric
            cooldown_period: Time (minutes) before the rule can trigger again
            notification_channels: Channels to notify when rule triggers
            description: Human-readable description of the rule
        """
        self.id = id
        self.metric_name = metric_name
        self.threshold = threshold
        self.direction = direction
        self.severity = severity
        self.evaluation_window = evaluation_window
        self.cooldown_period = cooldown_period
        self.notification_channels = notification_channels or ["log"]
        self.description = description
        
        # Tracking data
        self.last_triggered: Optional[datetime] = None
        self.trigger_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "direction": self.direction.value,
            "severity": self.severity.value,
            "evaluation_window": self.evaluation_window,
            "cooldown_period": self.cooldown_period,
            "notification_channels": self.notification_channels,
            "description": self.description,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackRule':
        """Create from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            RollbackRule: Created instance
        """
        rule = cls(
            id=data["id"],
            metric_name=data["metric_name"],
            threshold=data["threshold"],
            direction=ThresholdDirection(data["direction"]),
            severity=AlertSeverity(data["severity"]),
            evaluation_window=data.get("evaluation_window", 5),
            cooldown_period=data.get("cooldown_period", 60),
            notification_channels=data.get("notification_channels", ["log"]),
            description=data.get("description", "")
        )
        
        # Load tracking data
        if data.get("last_triggered"):
            rule.last_triggered = datetime.fromisoformat(data["last_triggered"])
        rule.trigger_count = data.get("trigger_count", 0)
        
        return rule
    
    def is_triggered(self, current_value: float) -> bool:
        """Check if the rule is triggered by the current metric value.
        
        Args:
            current_value: Current value of the metric
            
        Returns:
            bool: True if triggered, False otherwise
        """
        if self.direction == ThresholdDirection.ABOVE:
            return current_value > self.threshold
        else:
            return current_value < self.threshold
    
    def can_trigger(self) -> bool:
        """Check if the rule can trigger again (cooldown period elapsed).
        
        Returns:
            bool: True if can trigger, False if in cooldown
        """
        if self.last_triggered is None:
            return True
        
        cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_period)
        return datetime.now() > cooldown_end
    
    def record_trigger(self):
        """Record that the rule was triggered."""
        self.last_triggered = datetime.now()
        self.trigger_count += 1


class FlagRollbackConfig:
    """Configuration for when to roll back a feature flag."""
    
    def __init__(
        self,
        flag_id: str,
        rules: List[RollbackRule],
        auto_rollback: bool = True,
        require_multiple_triggers: bool = False,
        created_at: Optional[datetime] = None,
        modified_at: Optional[datetime] = None
    ):
        """Initialize a flag rollback configuration.
        
        Args:
            flag_id: ID of the feature flag
            rules: List of rollback rules
            auto_rollback: Whether to automatically roll back or just alert
            require_multiple_triggers: Whether to require multiple rules to trigger
            created_at: When the config was created
            modified_at: When the config was last modified
        """
        self.flag_id = flag_id
        self.rules = rules
        self.auto_rollback = auto_rollback
        self.require_multiple_triggers = require_multiple_triggers
        self.created_at = created_at or datetime.now()
        self.modified_at = modified_at or self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "flag_id": self.flag_id,
            "rules": [rule.to_dict() for rule in self.rules],
            "auto_rollback": self.auto_rollback,
            "require_multiple_triggers": self.require_multiple_triggers,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlagRollbackConfig':
        """Create from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            FlagRollbackConfig: Created instance
        """
        return cls(
            flag_id=data["flag_id"],
            rules=[RollbackRule.from_dict(rule_data) for rule_data in data["rules"]],
            auto_rollback=data.get("auto_rollback", True),
            require_multiple_triggers=data.get("require_multiple_triggers", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
            modified_at=datetime.fromisoformat(data["modified_at"]) if "modified_at" in data else None
        )
    
    def evaluate(
        self,
        metrics: Dict[str, float],
        min_rules_triggered: int = 1
    ) -> Tuple[bool, List[RollbackRule]]:
        """Evaluate the rules against the current metrics.
        
        Args:
            metrics: Current metric values
            min_rules_triggered: Minimum number of rules that must trigger
            
        Returns:
            Tuple[bool, List[RollbackRule]]: Whether to roll back and which rules triggered
        """
        triggered_rules = []
        
        for rule in self.rules:
            # Skip metrics that we don't have data for
            if rule.metric_name not in metrics:
                continue
            
            # Skip rules in cooldown
            if not rule.can_trigger():
                continue
            
            # Check if rule is triggered
            current_value = metrics[rule.metric_name]
            if rule.is_triggered(current_value):
                triggered_rules.append(rule)
                
                # Record the trigger
                rule.record_trigger()
        
        # Determine if we should roll back
        should_rollback = (
            len(triggered_rules) >= min_rules_triggered and
            (not self.require_multiple_triggers or len(triggered_rules) > 1)
        )
        
        return should_rollback, triggered_rules


class AutoRollbackService:
    """Service for automatically rolling back feature flags."""
    
    _instance = None
    
    def __init__(
        self,
        config_dir: str = "data/feature_flags/rollback",
        check_interval: int = 60,  # 1 minute
        notification_handlers: Optional[Dict[str, Callable[[str, AlertSeverity], None]]] = None
    ):
        """Initialize the auto-rollback service.
        
        Args:
            config_dir: Directory to store rollback configuration
            check_interval: How often to check metrics (seconds)
            notification_handlers: Handlers for different notification channels
        """
        self.config_dir = config_dir
        self.check_interval = check_interval
        self.notification_handlers = notification_handlers or {
            "log": lambda msg, severity: logger.warning(msg) if severity == AlertSeverity.WARNING else logger.critical(msg)
        }
        
        self.configs: Dict[str, FlagRollbackConfig] = {}
        self.rollback_history: List[Dict[str, Any]] = []
        
        # Create config directory
        os.makedirs(config_dir, exist_ok=True)
        
        # Background check thread
        self._stop_event = threading.Event()
        self._check_thread = None
        
        # Load configurations
        self._load_configs()
        
        # Start check thread
        self._start_check_thread()
        
        logger.info(f"Auto-rollback service initialized with {len(self.configs)} configurations")
    
    def _load_configs(self):
        """Load rollback configurations from disk."""
        try:
            config_file = os.path.join(self.config_dir, "rollback_configs.json")
            if not os.path.exists(config_file):
                return
            
            with open(config_file, 'r') as f:
                data = json.load(f)
                
                # Load configs
                for config_data in data.get("configs", []):
                    try:
                        config = FlagRollbackConfig.from_dict(config_data)
                        self.configs[config.flag_id] = config
                    except Exception as e:
                        logger.error(f"Error loading rollback config: {e}")
                
                # Load history
                self.rollback_history = data.get("history", [])
                
                # Clean up old history entries (keep last 100)
                if len(self.rollback_history) > 100:
                    self.rollback_history = self.rollback_history[-100:]
        
        except Exception as e:
            logger.error(f"Error loading rollback configurations: {e}")
    
    def _save_configs(self):
        """Save rollback configurations to disk."""
        try:
            config_file = os.path.join(self.config_dir, "rollback_configs.json")
            
            # Create backup if file exists
            if os.path.exists(config_file):
                backup_file = f"{config_file}.bak"
                os.replace(config_file, backup_file)
            
            # Save configs
            with open(config_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "configs": [config.to_dict() for config in self.configs.values()],
                    "history": self.rollback_history
                }, f, indent=2)
            
            logger.debug(f"Saved rollback configurations")
        
        except Exception as e:
            logger.error(f"Error saving rollback configurations: {e}")
    
    def _start_check_thread(self):
        """Start the background check thread."""
        if self._check_thread is not None and self._check_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._check_thread = threading.Thread(
            target=self._check_worker,
            daemon=True,
            name="AutoRollbackCheck"
        )
        self._check_thread.start()
        
        logger.info(f"Started auto-rollback check thread (interval={self.check_interval}s)")
    
    def _check_worker(self):
        """Worker function for the check thread."""
        while not self._stop_event.wait(self.check_interval):
            try:
                self.check_all_metrics()
            except Exception as e:
                logger.error(f"Error in auto-rollback check: {e}")
    
    def add_rollback_config(
        self,
        flag_id: str,
        rules: List[RollbackRule],
        auto_rollback: bool = True,
        require_multiple_triggers: bool = False
    ) -> Tuple[bool, str]:
        """Add a rollback configuration for a feature flag.
        
        Args:
            flag_id: ID of the feature flag
            rules: List of rollback rules
            auto_rollback: Whether to automatically roll back or just alert
            require_multiple_triggers: Whether to require multiple rules to trigger
            
        Returns:
            Tuple[bool, str]: Success and message
        """
        # Validate flag exists
        service = get_feature_flag_service()
        flag = service.get_flag(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        # Create config
        config = FlagRollbackConfig(
            flag_id=flag_id,
            rules=rules,
            auto_rollback=auto_rollback,
            require_multiple_triggers=require_multiple_triggers
        )
        
        # Add to configs
        self.configs[flag_id] = config
        
        # Save configs
        self._save_configs()
        
        return True, f"Added rollback configuration for flag '{flag_id}' with {len(rules)} rules"
    
    def remove_rollback_config(self, flag_id: str) -> Tuple[bool, str]:
        """Remove a rollback configuration for a feature flag.
        
        Args:
            flag_id: ID of the feature flag
            
        Returns:
            Tuple[bool, str]: Success and message
        """
        if flag_id not in self.configs:
            return False, f"No rollback configuration found for flag '{flag_id}'"
        
        # Remove config
        del self.configs[flag_id]
        
        # Save configs
        self._save_configs()
        
        return True, f"Removed rollback configuration for flag '{flag_id}'"
    
    def get_rollback_config(self, flag_id: str) -> Optional[FlagRollbackConfig]:
        """Get the rollback configuration for a feature flag.
        
        Args:
            flag_id: ID of the feature flag
            
        Returns:
            Optional[FlagRollbackConfig]: The configuration or None if not found
        """
        return self.configs.get(flag_id)
    
    def check_all_metrics(self):
        """Check all metrics for all configured flags."""
        if not self.configs:
            return
        
        # Get metrics collector
        metrics_collector = get_metrics_collector()
        
        # Get current metrics
        metrics = self._get_current_metrics()
        
        # Check each flag
        for flag_id, config in list(self.configs.items()):
            self.check_flag_metrics(flag_id, metrics)
    
    def check_flag_metrics(
        self,
        flag_id: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[RollbackRule]]:
        """Check metrics for a specific flag.
        
        Args:
            flag_id: ID of the feature flag
            metrics: Current metrics (optional)
            
        Returns:
            Tuple[bool, List[RollbackRule]]: Whether action was taken and triggered rules
        """
        # Get config for this flag
        config = self.get_rollback_config(flag_id)
        if not config:
            return False, []
        
        # Get current metrics if not provided
        if metrics is None:
            metrics = self._get_current_metrics()
        
        # Get feature flag service
        service = get_feature_flag_service()
        
        # Check if flag is enabled
        if not service.is_enabled(flag_id):
            return False, []  # No need to roll back if not enabled
        
        # Evaluate rules
        should_rollback, triggered_rules = config.evaluate(
            metrics,
            min_rules_triggered=2 if config.require_multiple_triggers else 1
        )
        
        # Take action if needed
        if should_rollback:
            if config.auto_rollback:
                # Roll back the flag
                success, message = service.set_flag(
                    flag_id=flag_id,
                    enabled=False,
                    changed_by="auto_rollback",
                    reason=f"Auto rollback triggered by {len(triggered_rules)} rules"
                )
                
                if success:
                    logger.warning(f"Auto-rolled back flag '{flag_id}' due to rule violations")
                    
                    # Record rollback
                    self._record_rollback(flag_id, triggered_rules, metrics, success=True)
                    
                    # Send notifications
                    self._send_notifications(
                        flag_id=flag_id,
                        rules=triggered_rules,
                        metrics=metrics,
                        action="ROLLBACK"
                    )
                else:
                    logger.error(f"Failed to auto-roll back flag '{flag_id}': {message}")
                    
                    # Record failed rollback
                    self._record_rollback(flag_id, triggered_rules, metrics, success=False, error=message)
            else:
                # Just send notifications
                logger.warning(f"Rollback conditions met for flag '{flag_id}' but auto-rollback is disabled")
                
                # Send notifications
                self._send_notifications(
                    flag_id=flag_id,
                    rules=triggered_rules,
                    metrics=metrics,
                    action="ALERT"
                )
                
                # Record alert
                self._record_rollback(
                    flag_id=flag_id,
                    rules=triggered_rules,
                    metrics=metrics,
                    success=False,
                    auto_rollback=False
                )
                
            return True, triggered_rules
        
        return False, []
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get the current performance metrics.
        
        Returns:
            Dict[str, float]: Current metrics
        """
        metrics_collector = get_metrics_collector()
        
        # Get most recent snapshot
        snapshots = metrics_collector.performance_snapshots
        if not snapshots:
            return {}
        
        # Sort by timestamp (newest first)
        sorted_snapshots = sorted(
            snapshots,
            key=lambda s: s["timestamp"],
            reverse=True
        )
        
        # Get metrics from most recent snapshot
        return sorted_snapshots[0]["performance"]
    
    def _send_notifications(
        self,
        flag_id: str,
        rules: List[RollbackRule],
        metrics: Dict[str, float],
        action: str
    ):
        """Send notifications for triggered rules.
        
        Args:
            flag_id: ID of the feature flag
            rules: Triggered rules
            metrics: Current metrics
            action: Action taken (ALERT or ROLLBACK)
        """
        # Get highest severity
        severity = max(rule.severity for rule in rules)
        
        # Build message
        message = f"{action}: Flag '{flag_id}' triggered {len(rules)} rules:\n"
        
        for rule in rules:
            metric_value = metrics.get(rule.metric_name, "N/A")
            message += (f"  - {rule.metric_name}: {metric_value} "
                        f"{'>' if rule.direction == ThresholdDirection.ABOVE else '<'} "
                        f"{rule.threshold}\n")
        
        # Send to each notification channel
        for rule in rules:
            for channel in rule.notification_channels:
                if channel in self.notification_handlers:
                    try:
                        self.notification_handlers[channel](message, severity)
                    except Exception as e:
                        logger.error(f"Error sending notification to {channel}: {e}")
    
    def _record_rollback(
        self,
        flag_id: str,
        rules: List[RollbackRule],
        metrics: Dict[str, float],
        success: bool,
        error: Optional[str] = None,
        auto_rollback: bool = True
    ):
        """Record a rollback or alert event.
        
        Args:
            flag_id: ID of the feature flag
            rules: Triggered rules
            metrics: Current metrics
            success: Whether the rollback was successful
            error: Error message if unsuccessful
            auto_rollback: Whether auto-rollback was enabled
        """
        # Create record
        record = {
            "timestamp": datetime.now().isoformat(),
            "flag_id": flag_id,
            "rules_triggered": [rule.id for rule in rules],
            "metrics": {k: v for k, v in metrics.items() if any(r.metric_name == k for r in rules)},
            "action": "ROLLBACK" if auto_rollback else "ALERT",
            "success": success
        }
        
        if error:
            record["error"] = error
        
        # Add to history
        self.rollback_history.append(record)
        
        # Save configs
        self._save_configs()
    
    def cleanup(self):
        """Clean up resources used by the service."""
        # Stop check thread
        if self._check_thread and self._check_thread.is_alive():
            self._stop_event.set()
            self._check_thread.join(timeout=5)
        
        # Save configs
        self._save_configs()


def get_auto_rollback_service() -> AutoRollbackService:
    """Get the singleton auto-rollback service.
    
    Returns:
        AutoRollbackService: The auto-rollback service
    """
    if AutoRollbackService._instance is None:
        AutoRollbackService._instance = AutoRollbackService()
    
    return AutoRollbackService._instance 