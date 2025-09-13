#!/usr/bin/env python3
"""
Automated Rollback System for Feature Flags

This module provides automatic rollback capabilities for feature flags when predefined
alert conditions are met. It continuously monitors system metrics and can automatically
disable problematic flags to maintain system stability during market volatility.
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
import pandas as pd
import numpy as np
from enum import Enum

from .metrics_integration import FeatureFlagMetrics

# Setup logging
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Severity levels for alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Status of an alert"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

class RollbackStrategy(Enum):
    """Strategy to use when rolling back a feature flag"""
    DISABLE_IMMEDIATELY = "disable_immediately"
    GRADUAL_ROLLBACK = "gradual_rollback"
    NOTIFY_ONLY = "notify_only"

class AlertCondition:
    """Defines a condition that will trigger an alert"""
    
    def __init__(self, 
                name: str,
                metric_name: str,
                threshold: float,
                comparison_operator: str = "greater_than",
                duration_seconds: int = 60,
                severity: AlertSeverity = AlertSeverity.WARNING):
        """
        Initialize an alert condition.
        
        Args:
            name: Name of the alert condition
            metric_name: Name of the metric to monitor
            threshold: Threshold value that triggers the alert
            comparison_operator: One of: greater_than, less_than, equals, not_equals
            duration_seconds: Duration the condition must be true before alerting
            severity: Severity level of the alert
        """
        self.name = name
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison_operator = comparison_operator
        self.duration_seconds = duration_seconds
        self.severity = severity
        
        # Validate operator
        valid_operators = ["greater_than", "less_than", "equals", "not_equals"]
        if comparison_operator not in valid_operators:
            raise ValueError(f"Invalid comparison operator. Must be one of: {valid_operators}")
    
    def check(self, value: float) -> bool:
        """
        Check if the value triggers the alert condition.
        
        Args:
            value: Current value of the metric
            
        Returns:
            bool: True if the condition is triggered, False otherwise
        """
        if self.comparison_operator == "greater_than":
            return value > self.threshold
        elif self.comparison_operator == "less_than":
            return value < self.threshold
        elif self.comparison_operator == "equals":
            return value == self.threshold
        elif self.comparison_operator == "not_equals":
            return value != self.threshold
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "comparison_operator": self.comparison_operator,
            "duration_seconds": self.duration_seconds,
            "severity": self.severity.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCondition':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            metric_name=data["metric_name"],
            threshold=data["threshold"],
            comparison_operator=data["comparison_operator"],
            duration_seconds=data["duration_seconds"],
            severity=AlertSeverity(data["severity"])
        )

class Alert:
    """Represents an alert triggered by a condition"""
    
    def __init__(self,
                condition: AlertCondition,
                flag_name: str,
                current_value: float,
                created_at: Optional[datetime] = None):
        """
        Initialize an alert.
        
        Args:
            condition: The condition that triggered the alert
            flag_name: Name of the feature flag related to the alert
            current_value: Current value that triggered the alert
            created_at: When the alert was created (defaults to now)
        """
        self.id = f"{int(time.time())}_{flag_name}_{condition.name}"
        self.condition = condition
        self.flag_name = flag_name
        self.current_value = current_value
        self.created_at = created_at or datetime.now()
        self.status = AlertStatus.ACTIVE
        self.resolved_at = None
        self.acknowledged_at = None
        self.acknowledged_by = None
        self.notes = []
    
    def acknowledge(self, user: str, note: Optional[str] = None):
        """
        Acknowledge the alert.
        
        Args:
            user: User acknowledging the alert
            note: Optional note about the acknowledgment
        """
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = user
        
        if note:
            self.add_note(user, note)
    
    def resolve(self, user: str, note: Optional[str] = None):
        """
        Resolve the alert.
        
        Args:
            user: User resolving the alert
            note: Optional note about the resolution
        """
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        
        if note:
            self.add_note(user, note)
    
    def add_note(self, user: str, note: str):
        """
        Add a note to the alert.
        
        Args:
            user: User adding the note
            note: Note text
        """
        self.notes.append({
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "text": note
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "condition": self.condition.to_dict(),
            "flag_name": self.flag_name,
            "current_value": self.current_value,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary"""
        alert = cls(
            condition=AlertCondition.from_dict(data["condition"]),
            flag_name=data["flag_name"],
            current_value=data["current_value"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
        
        alert.id = data["id"]
        alert.status = AlertStatus(data["status"])
        
        if data.get("resolved_at"):
            alert.resolved_at = datetime.fromisoformat(data["resolved_at"])
            
        if data.get("acknowledged_at"):
            alert.acknowledged_at = datetime.fromisoformat(data["acknowledged_at"])
            alert.acknowledged_by = data.get("acknowledged_by")
            
        alert.notes = data.get("notes", [])
        
        return alert

class RollbackRule:
    """Defines a rule for automatic feature flag rollback"""
    
    def __init__(self,
                name: str,
                flag_name: str,
                conditions: List[AlertCondition],
                strategy: RollbackStrategy = RollbackStrategy.DISABLE_IMMEDIATELY,
                require_all_conditions: bool = False,
                cooldown_minutes: int = 60,
                notify_users: Optional[List[str]] = None):
        """
        Initialize a rollback rule.
        
        Args:
            name: Name of the rollback rule
            flag_name: Name of the feature flag
            conditions: List of alert conditions that trigger the rollback
            strategy: Strategy to use when rolling back
            require_all_conditions: If True, all conditions must be met to trigger
            cooldown_minutes: Minimum time between rollbacks
            notify_users: List of users to notify on rollback
        """
        self.name = name
        self.flag_name = flag_name
        self.conditions = conditions
        self.strategy = strategy
        self.require_all_conditions = require_all_conditions
        self.cooldown_minutes = cooldown_minutes
        self.notify_users = notify_users or []
        
        # Track rollback history
        self.last_rollback = None
        self.rollback_count = 0
    
    def should_rollback(self, metrics: Dict[str, float]) -> Tuple[bool, List[AlertCondition]]:
        """
        Check if the rule should trigger a rollback based on current metrics.
        
        Args:
            metrics: Dictionary of current metric values
            
        Returns:
            Tuple[bool, List[AlertCondition]]: Whether to rollback and which conditions triggered
        """
        # Check cooldown period
        if self.last_rollback:
            cooldown_end = self.last_rollback + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False, []
        
        # Check conditions
        triggered_conditions = []
        
        for condition in self.conditions:
            if condition.metric_name in metrics:
                value = metrics[condition.metric_name]
                if condition.check(value):
                    triggered_conditions.append(condition)
        
        # Determine if we should rollback
        should_rollback = False
        
        if self.require_all_conditions:
            # All conditions must be met
            should_rollback = len(triggered_conditions) == len(self.conditions)
        else:
            # Any condition can trigger
            should_rollback = len(triggered_conditions) > 0
        
        return should_rollback, triggered_conditions
    
    def record_rollback(self):
        """Record that a rollback occurred"""
        self.last_rollback = datetime.now()
        self.rollback_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "flag_name": self.flag_name,
            "conditions": [c.to_dict() for c in self.conditions],
            "strategy": self.strategy.value,
            "require_all_conditions": self.require_all_conditions,
            "cooldown_minutes": self.cooldown_minutes,
            "notify_users": self.notify_users,
            "last_rollback": self.last_rollback.isoformat() if self.last_rollback else None,
            "rollback_count": self.rollback_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackRule':
        """Create from dictionary"""
        rule = cls(
            name=data["name"],
            flag_name=data["flag_name"],
            conditions=[AlertCondition.from_dict(c) for c in data["conditions"]],
            strategy=RollbackStrategy(data["strategy"]),
            require_all_conditions=data["require_all_conditions"],
            cooldown_minutes=data["cooldown_minutes"],
            notify_users=data["notify_users"]
        )
        
        if data.get("last_rollback"):
            rule.last_rollback = datetime.fromisoformat(data["last_rollback"])
            
        rule.rollback_count = data.get("rollback_count", 0)
        
        return rule

class AutomatedRollbackSystem:
    """
    System for automatically rolling back feature flags based on alert conditions.
    
    Monitors system metrics in real-time and can automatically disable feature flags
    when predefined alert conditions are met to maintain system stability.
    """
    
    def __init__(self,
                storage_path: str = "data/feature_flags/rollbacks",
                metrics_instance: Optional[FeatureFlagMetrics] = None,
                flag_service = None,
                notification_callbacks: Optional[List[Callable]] = None,
                polling_interval_seconds: int = 60):
        """
        Initialize the automated rollback system.
        
        Args:
            storage_path: Path to store rollback data
            metrics_instance: FeatureFlagMetrics instance
            flag_service: Feature flag service to control flags
            notification_callbacks: List of functions to call for notifications
            polling_interval_seconds: How often to check metrics
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Store feature flag service reference
        self.flag_service = flag_service
        
        # Initialize metrics
        if metrics_instance:
            self.metrics = metrics_instance
        else:
            from .metrics_integration import FeatureFlagMetrics
            self.metrics = FeatureFlagMetrics()
        
        # Alert and rollback tracking
        self.rules: Dict[str, RollbackRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.notification_callbacks = notification_callbacks or []
        
        # Monitoring thread settings
        self.polling_interval = polling_interval_seconds
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Metrics cache
        self._metrics_cache = {}
        self._last_metrics_update = None
        
        # Load existing rules and alerts
        self._load_rules()
        self._load_alerts()
        
        logger.info(f"Initialized automated rollback system with {len(self.rules)} rules")
    
    def _load_rules(self):
        """Load existing rollback rules from storage"""
        try:
            rules_file = os.path.join(self.storage_path, "rollback_rules.json")
            if not os.path.exists(rules_file):
                return
                
            with open(rules_file, 'r') as f:
                data = json.load(f)
                
            for rule_data in data.get("rules", []):
                try:
                    rule = RollbackRule.from_dict(rule_data)
                    self.rules[rule.name] = rule
                except Exception as e:
                    logger.error(f"Error loading rule {rule_data.get('name')}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading rollback rules: {str(e)}")
    
    def _save_rules(self):
        """Save rollback rules to storage"""
        try:
            rules_file = os.path.join(self.storage_path, "rollback_rules.json")
            
            with open(rules_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "rules": [rule.to_dict() for rule in self.rules.values()]
                }, f, indent=2)
                
            logger.debug(f"Saved {len(self.rules)} rollback rules")
            
        except Exception as e:
            logger.error(f"Error saving rollback rules: {str(e)}")
    
    def _load_alerts(self):
        """Load existing alerts from storage"""
        try:
            alerts_file = os.path.join(self.storage_path, "alerts.json")
            if not os.path.exists(alerts_file):
                return
                
            with open(alerts_file, 'r') as f:
                data = json.load(f)
                
            for alert_data in data.get("alerts", []):
                try:
                    alert = Alert.from_dict(alert_data)
                    self.alerts[alert.id] = alert
                except Exception as e:
                    logger.error(f"Error loading alert {alert_data.get('id')}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading alerts: {str(e)}")
    
    def _save_alerts(self):
        """Save alerts to storage"""
        try:
            alerts_file = os.path.join(self.storage_path, "alerts.json")
            
            with open(alerts_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "alerts": [alert.to_dict() for alert in self.alerts.values()]
                }, f, indent=2)
                
            logger.debug(f"Saved {len(self.alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Error saving alerts: {str(e)}")
    
    def add_rule(self, rule: RollbackRule) -> bool:
        """
        Add a new rollback rule.
        
        Args:
            rule: The rollback rule to add
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        if rule.name in self.rules:
            logger.warning(f"Rule '{rule.name}' already exists")
            return False
            
        self.rules[rule.name] = rule
        self._save_rules()
        
        logger.info(f"Added rollback rule '{rule.name}' for flag '{rule.flag_name}'")
        return True
    
    def update_rule(self, rule: RollbackRule) -> bool:
        """
        Update an existing rollback rule.
        
        Args:
            rule: The rollback rule to update
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        if rule.name not in self.rules:
            logger.warning(f"Rule '{rule.name}' does not exist")
            return False
            
        # Preserve rollback history
        rule.last_rollback = self.rules[rule.name].last_rollback
        rule.rollback_count = self.rules[rule.name].rollback_count
        
        self.rules[rule.name] = rule
        self._save_rules()
        
        logger.info(f"Updated rollback rule '{rule.name}'")
        return True
    
    def delete_rule(self, rule_name: str) -> bool:
        """
        Delete a rollback rule.
        
        Args:
            rule_name: Name of the rule to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if rule_name not in self.rules:
            logger.warning(f"Rule '{rule_name}' does not exist")
            return False
            
        del self.rules[rule_name]
        self._save_rules()
        
        logger.info(f"Deleted rollback rule '{rule_name}'")
        return True
    
    def get_rule(self, rule_name: str) -> Optional[RollbackRule]:
        """
        Get a rollback rule by name.
        
        Args:
            rule_name: Name of the rule to get
            
        Returns:
            Optional[RollbackRule]: The rule, or None if not found
        """
        return self.rules.get(rule_name)
    
    def get_rules_for_flag(self, flag_name: str) -> List[RollbackRule]:
        """
        Get all rules for a specific flag.
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            List[RollbackRule]: List of rules for the flag
        """
        return [rule for rule in self.rules.values() if rule.flag_name == flag_name]
    
    def create_alert(self, condition: AlertCondition, flag_name: str, current_value: float) -> Alert:
        """
        Create a new alert.
        
        Args:
            condition: The condition that triggered the alert
            flag_name: Name of the feature flag
            current_value: Current value that triggered the alert
            
        Returns:
            Alert: The created alert
        """
        alert = Alert(condition, flag_name, current_value)
        self.alerts[alert.id] = alert
        self._save_alerts()
        
        # Send notifications
        self._notify_alert(alert)
        
        logger.info(f"Created alert '{alert.id}' for flag '{flag_name}'")
        return alert
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active alerts.
        
        Returns:
            List[Alert]: List of active alerts
        """
        return [alert for alert in self.alerts.values() 
                if alert.status == AlertStatus.ACTIVE]
    
    def get_alerts_for_flag(self, flag_name: str) -> List[Alert]:
        """
        Get all alerts for a specific flag.
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            List[Alert]: List of alerts for the flag
        """
        return [alert for alert in self.alerts.values() if alert.flag_name == flag_name]
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Started monitoring thread")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=10)
        
        logger.info("Stopped monitoring thread")
    
    def _monitoring_loop(self):
        """Background thread that periodically checks metrics"""
        logger.info("Monitoring loop started")
        
        while not self._stop_monitoring.is_set():
            try:
                # Update metrics cache
                self._update_metrics_cache()
                
                # Check rules
                self._check_rules()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
            # Wait for next interval
            self._stop_monitoring.wait(self.polling_interval)
    
    def _update_metrics_cache(self):
        """Update the cache of current metric values"""
        try:
            # Get metrics from the metrics system
            if self.metrics:
                # Check if we need to update (avoid too frequent updates)
                now = datetime.now()
                if (self._last_metrics_update is None or 
                    (now - self._last_metrics_update).total_seconds() >= self.polling_interval):
                    
                    # Get system metrics
                    self._metrics_cache = self.metrics.get_current_metrics()
                    self._last_metrics_update = now
                    
                    logger.debug(f"Updated metrics cache with {len(self._metrics_cache)} metrics")
            
        except Exception as e:
            logger.error(f"Error updating metrics cache: {str(e)}")
    
    def _check_rules(self):
        """Check all rules against current metrics"""
        # Skip if no metrics available
        if not self._metrics_cache:
            return
            
        for rule_name, rule in self.rules.items():
            try:
                # Check if rule should trigger
                should_rollback, triggered_conditions = rule.should_rollback(self._metrics_cache)
                
                if should_rollback:
                    # Create alerts for triggered conditions
                    alerts = []
                    for condition in triggered_conditions:
                        current_value = self._metrics_cache.get(condition.metric_name, 0)
                        alert = self.create_alert(condition, rule.flag_name, current_value)
                        alerts.append(alert)
                    
                    # Perform rollback
                    self._rollback_flag(rule, alerts)
            
            except Exception as e:
                logger.error(f"Error checking rule '{rule_name}': {str(e)}")
    
    def _rollback_flag(self, rule: RollbackRule, alerts: List[Alert]):
        """
        Rollback a feature flag based on the specified strategy.
        
        Args:
            rule: The rollback rule
            alerts: List of alerts that triggered the rollback
        """
        logger.warning(f"Rolling back flag '{rule.flag_name}' due to rule '{rule.name}'")
        
        # Record the rollback
        rule.record_rollback()
        self._save_rules()
        
        # Get the feature flag
        if not self.flag_service:
            logger.error("No feature flag service available for rollback")
            return
            
        # Apply rollback strategy
        try:
            if rule.strategy == RollbackStrategy.DISABLE_IMMEDIATELY:
                # Disable the flag immediately
                self.flag_service.disable_flag(rule.flag_name)
                logger.info(f"Disabled flag '{rule.flag_name}' due to rule '{rule.name}'")
                
            elif rule.strategy == RollbackStrategy.GRADUAL_ROLLBACK:
                # Gradually roll back by reducing percentage
                self.flag_service.set_flag_percentage(rule.flag_name, 0)
                logger.info(f"Gradually rolling back flag '{rule.flag_name}' due to rule '{rule.name}'")
                
            # Create rollback record
            rollback_record = {
                "timestamp": datetime.now().isoformat(),
                "rule": rule.name,
                "flag_name": rule.flag_name,
                "strategy": rule.strategy.value,
                "alerts": [alert.id for alert in alerts],
                "metrics": {k: v for k, v in self._metrics_cache.items()}
            }
            
            # Save rollback record
            self._save_rollback_record(rollback_record)
            
            # Send notifications
            self._notify_rollback(rule, alerts)
            
        except Exception as e:
            logger.error(f"Error rolling back flag '{rule.flag_name}': {str(e)}")
    
    def _save_rollback_record(self, record: Dict[str, Any]):
        """
        Save a rollback record.
        
        Args:
            record: Rollback information to save
        """
        try:
            # Create rollbacks directory if needed
            rollbacks_dir = os.path.join(self.storage_path, "rollbacks")
            os.makedirs(rollbacks_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rollback_{timestamp}_{record['flag_name']}.json"
            file_path = os.path.join(rollbacks_dir, filename)
            
            # Save the record
            with open(file_path, 'w') as f:
                json.dump(record, f, indent=2)
                
            logger.debug(f"Saved rollback record to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving rollback record: {str(e)}")
    
    def _notify_alert(self, alert: Alert):
        """
        Send notifications about a new alert.
        
        Args:
            alert: The alert that was created
        """
        # Format alert message
        message = f"ALERT: {alert.condition.severity.value.upper()} - {alert.condition.name}\n"
        message += f"Flag: {alert.flag_name}\n"
        message += f"Metric: {alert.condition.metric_name} = {alert.current_value} "
        message += f"({alert.condition.comparison_operator} {alert.condition.threshold})\n"
        message += f"Time: {alert.created_at.isoformat()}"
        
        # Call notification callbacks
        for callback in self.notification_callbacks:
            try:
                callback("alert", message, alert.to_dict())
            except Exception as e:
                logger.error(f"Error sending alert notification: {str(e)}")
    
    def _notify_rollback(self, rule: RollbackRule, alerts: List[Alert]):
        """
        Send notifications about a rollback.
        
        Args:
            rule: The rule that triggered the rollback
            alerts: The alerts that triggered the rollback
        """
        # Format rollback message
        message = f"ROLLBACK: Flag '{rule.flag_name}' rolled back due to rule '{rule.name}'\n"
        message += f"Strategy: {rule.strategy.value}\n"
        message += f"Time: {datetime.now().isoformat()}\n\n"
        message += "Triggered by alerts:\n"
        
        for alert in alerts:
            message += f"- {alert.condition.name}: {alert.condition.metric_name} = {alert.current_value}\n"
        
        # Call notification callbacks
        for callback in self.notification_callbacks:
            try:
                callback("rollback", message, {
                    "rule": rule.to_dict(),
                    "alerts": [alert.to_dict() for alert in alerts]
                })
            except Exception as e:
                logger.error(f"Error sending rollback notification: {str(e)}")
    
    def get_rollback_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get rollback history for the specified number of days.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of rollback records
        """
        rollbacks = []
        
        try:
            # Get rollbacks directory
            rollbacks_dir = os.path.join(self.storage_path, "rollbacks")
            if not os.path.exists(rollbacks_dir):
                return rollbacks
                
            # Calculate cutoff date
            cutoff = datetime.now() - timedelta(days=days)
            
            # List rollback files
            for filename in os.listdir(rollbacks_dir):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(rollbacks_dir, filename)
                
                # Check file modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff:
                    continue
                
                # Load rollback record
                with open(file_path, 'r') as f:
                    record = json.load(f)
                    rollbacks.append(record)
                    
            # Sort by timestamp (newest first)
            rollbacks.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting rollback history: {str(e)}")
            
        return rollbacks

    def create_default_rules(self) -> List[str]:
        """
        Create a set of default rollback rules.
        
        Returns:
            List[str]: Names of created rules
        """
        created_rules = []
        
        # Common metrics to monitor
        metrics = {
            "error_rate": 0.05,  # 5% error rate threshold
            "latency_ms": 500,   # 500ms latency threshold
            "memory_usage_mb": 2000,  # 2GB memory threshold
            "cpu_usage_percent": 80,  # 80% CPU threshold
            "trading_errors": 3,  # 3 trading errors threshold
            "drawdown_percent": 5,  # 5% drawdown threshold
        }
        
        # Get all feature flags
        flags = []
        if self.flag_service:
            flags = self.flag_service.get_all_flags()
        
        # Create rules for critical flags
        for flag in flags:
            flag_name = flag.name if hasattr(flag, 'name') else flag
            
            # Create error rate rule
            error_condition = AlertCondition(
                name=f"{flag_name}_error_rate",
                metric_name="error_rate",
                threshold=metrics["error_rate"],
                comparison_operator="greater_than",
                severity=AlertSeverity.CRITICAL
            )
            
            error_rule = RollbackRule(
                name=f"{flag_name}_error_rollback",
                flag_name=flag_name,
                conditions=[error_condition],
                strategy=RollbackStrategy.DISABLE_IMMEDIATELY,
                cooldown_minutes=60
            )
            
            if self.add_rule(error_rule):
                created_rules.append(error_rule.name)
            
            # Create latency rule
            latency_condition = AlertCondition(
                name=f"{flag_name}_high_latency",
                metric_name="latency_ms",
                threshold=metrics["latency_ms"],
                comparison_operator="greater_than",
                severity=AlertSeverity.WARNING
            )
            
            latency_rule = RollbackRule(
                name=f"{flag_name}_latency_rollback",
                flag_name=flag_name,
                conditions=[latency_condition],
                strategy=RollbackStrategy.GRADUAL_ROLLBACK,
                cooldown_minutes=30
            )
            
            if self.add_rule(latency_rule):
                created_rules.append(latency_rule.name)
        
        return created_rules

# Initialize the system
def init_rollback_system(
    flag_service=None, 
    metrics_instance=None, 
    notification_callbacks=None
) -> AutomatedRollbackSystem:
    """
    Initialize the automated rollback system.
    
    Args:
        flag_service: Feature flag service instance
        metrics_instance: Metrics tracking instance
        notification_callbacks: List of notification callbacks
        
    Returns:
        AutomatedRollbackSystem: Initialized system
    """
    system = AutomatedRollbackSystem(
        flag_service=flag_service,
        metrics_instance=metrics_instance,
        notification_callbacks=notification_callbacks
    )
    
    # Start monitoring thread
    system.start_monitoring()
    
    return system 