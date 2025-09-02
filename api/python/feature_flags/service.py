#!/usr/bin/env python3
"""
Feature Flag Service

Provides a centralized service for managing feature flags, which allows
for selective enabling/disabling of trading strategies or risk features
without requiring a full deployment.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import hashlib
import re

logger = logging.getLogger(__name__)

class FlagCategory(Enum):
    """Categories for feature flags to organize them by purpose."""
    STRATEGY = auto()
    RISK = auto()
    MONITORING = auto()
    NOTIFICATION = auto()
    DATA = auto()
    EXECUTION = auto()
    EXPERIMENTAL = auto()
    SYSTEM = auto()

class AssetClass(Enum):
    """Supported asset classes for targeted feature rollouts."""
    ALL = auto()
    FOREX = auto()
    CRYPTO = auto()
    EQUITY = auto()
    FUTURES = auto()
    OPTIONS = auto()
    INDICES = auto()
    COMMODITIES = auto()

@dataclass
class FlagChangeEvent:
    """Represents a change to a feature flag."""
    flag_id: str
    enabled: bool
    timestamp: datetime = field(default_factory=datetime.now)
    changed_by: str = "system"
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary for serialization."""
        return {
            "flag_id": self.flag_id,
            "enabled": self.enabled,
            "timestamp": self.timestamp.isoformat(),
            "changed_by": self.changed_by,
            "reason": self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlagChangeEvent':
        """Create a FlagChangeEvent from a dictionary."""
        return cls(
            flag_id=data["flag_id"],
            enabled=data["enabled"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            changed_by=data["changed_by"],
            reason=data.get("reason")
        )

@dataclass
class ContextRule:
    """A rule for evaluating whether a feature flag should be enabled in a given context."""
    rule_type: str
    parameters: Dict[str, Any]
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the rule against the provided context.
        
        Args:
            context: The context to evaluate against
            
        Returns:
            bool: True if the rule passes, False otherwise
        """
        if self.rule_type == "asset_class":
            return self._evaluate_asset_class(context)
        elif self.rule_type == "time_window":
            return self._evaluate_time_window(context)
        elif self.rule_type == "account_value":
            return self._evaluate_account_value(context)
        elif self.rule_type == "market_condition":
            return self._evaluate_market_condition(context)
        else:
            logger.warning(f"Unknown rule type: {self.rule_type}")
            return True  # Default to enabled for unknown rules
    
    def _evaluate_asset_class(self, context: Dict[str, Any]) -> bool:
        """Check if the asset class matches."""
        if "asset_class" not in context:
            return True  # No asset class specified, pass by default
        
        allowed_classes = self.parameters.get("asset_classes", [])
        if "ALL" in allowed_classes:
            return True
        
        return context["asset_class"] in allowed_classes
    
    def _evaluate_time_window(self, context: Dict[str, Any]) -> bool:
        """Check if current time is within specified window."""
        if "current_time" not in context:
            return True  # No time specified, pass by default
        
        current_time = context["current_time"]
        start_time = self.parameters.get("start_time")
        end_time = self.parameters.get("end_time")
        
        if start_time and end_time:
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()
            current = current_time.time()
            
            # Handle cases where the window spans midnight
            if start <= end:
                return start <= current <= end
            else:
                return start <= current or current <= end
        
        return True
    
    def _evaluate_account_value(self, context: Dict[str, Any]) -> bool:
        """Check if account value is within range."""
        if "account_value" not in context:
            return True  # No account value specified, pass by default
        
        account_value = context["account_value"]
        min_value = self.parameters.get("min_value")
        max_value = self.parameters.get("max_value")
        
        if min_value is not None and account_value < min_value:
            return False
        
        if max_value is not None and account_value > max_value:
            return False
        
        return True
    
    def _evaluate_market_condition(self, context: Dict[str, Any]) -> bool:
        """Check market conditions."""
        if "market_condition" not in context:
            return True  # No market condition specified, pass by default
        
        condition = context["market_condition"]
        allowed_conditions = self.parameters.get("conditions", [])
        
        if not allowed_conditions:
            return True  # No conditions specified, pass by default
        
        return condition in allowed_conditions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_type": self.rule_type,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextRule':
        """Create from dictionary."""
        return cls(
            rule_type=data["rule_type"],
            parameters=data["parameters"]
        )

@dataclass
class FeatureFlag:
    """Represents a feature flag in the system."""
    id: str
    name: str
    description: str
    category: FlagCategory
    enabled: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    history: List[FlagChangeEvent] = field(default_factory=list)
    default: bool = False
    requires_confirmation: bool = False
    rollback_after_seconds: Optional[int] = None
    dependent_flags: Set[str] = field(default_factory=set)
    # New fields for gradual rollout
    rollout_percentage: int = 100  # 0-100%
    applicable_asset_classes: Set[str] = field(default_factory=lambda: {"ALL"})
    context_rules: List[ContextRule] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the flag to a dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.name,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "history": [event.to_dict() for event in self.history],
            "default": self.default,
            "requires_confirmation": self.requires_confirmation,
            "rollback_after_seconds": self.rollback_after_seconds,
            "dependent_flags": list(self.dependent_flags),
            # Add new fields
            "rollout_percentage": self.rollout_percentage,
            "applicable_asset_classes": list(self.applicable_asset_classes),
            "context_rules": [rule.to_dict() for rule in self.context_rules]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureFlag':
        """Create a FeatureFlag from a dictionary."""
        flag = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=FlagCategory[data["category"]],
            enabled=data["enabled"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            default=data.get("default", False),
            requires_confirmation=data.get("requires_confirmation", False),
            rollback_after_seconds=data.get("rollback_after_seconds"),
            dependent_flags=set(data.get("dependent_flags", [])),
            # Load new fields with defaults
            rollout_percentage=data.get("rollout_percentage", 100),
            applicable_asset_classes=set(data.get("applicable_asset_classes", ["ALL"]))
        )
        
        if "history" in data:
            flag.history = [FlagChangeEvent.from_dict(event) for event in data["history"]]
        
        # Load context rules
        if "context_rules" in data:
            flag.context_rules = [ContextRule.from_dict(rule) for rule in data["context_rules"]]
        
        return flag

# Type for flag change callbacks
FlagChangeCallback = Callable[[FlagChangeEvent], None]

class FeatureFlagService:
    """Service for managing feature flags."""
    _instance = None
    _instance_lock = threading.Lock()
    
    def __init__(
        self, 
        storage_dir: str = "data/feature_flags",
        flags_file: str = "flags.json",
        auto_save: bool = True
    ):
        """Initialize the feature flag service.
        
        Args:
            storage_dir: Directory to store feature flag data
            flags_file: Name of the file to store feature flags
            auto_save: Whether to automatically save flags on changes
        """
        self.storage_dir = storage_dir
        self.flags_file = flags_file
        self.auto_save = auto_save
        self.flags: Dict[str, FeatureFlag] = {}
        self.callbacks: List[FlagChangeCallback] = []
        self._rollback_timers: Dict[str, threading.Timer] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing flags from storage
        self._load_flags()
        
        # Start background save thread if auto_save is enabled
        self._save_thread = None
        self._save_stop_event = threading.Event()
        if auto_save:
            self._start_auto_save()
    
    def _start_auto_save(self):
        """Start the background auto-save thread."""
        def auto_save_worker():
            while not self._save_stop_event.wait(60):  # Save every minute
                self.save()
        
        self._save_thread = threading.Thread(
            target=auto_save_worker, 
            daemon=True,
            name="FeatureFlagAutoSave"
        )
        self._save_thread.start()
    
    def _load_flags(self):
        """Load feature flags from storage."""
        file_path = os.path.join(self.storage_dir, self.flags_file)
        if not os.path.exists(file_path):
            logger.info(f"Feature flags file not found at {file_path}, creating default flags")
            self._create_default_flags()
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for flag_data in data.get("flags", []):
                    flag = FeatureFlag.from_dict(flag_data)
                    self.flags[flag.id] = flag
                    
                    # Set up rollback timers for recently enabled flags with rollback
                    if flag.enabled and flag.rollback_after_seconds:
                        self._setup_rollback_timer(flag.id)
            
            logger.info(f"Loaded {len(self.flags)} feature flags from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
            # Create default flags if loading fails
            self._create_default_flags()
    
    def _create_default_flags(self):
        """Create default feature flags."""
        default_flags = [
            FeatureFlag(
                id="telegram_control",
                name="Telegram Control",
                description="Enable control of the trading bot via Telegram commands",
                category=FlagCategory.SYSTEM,
                enabled=True,
                default=True
            ),
            FeatureFlag(
                id="risk_limits",
                name="Risk Limits",
                description="Enable risk management limits to prevent excessive losses",
                category=FlagCategory.RISK,
                enabled=True,
                default=True,
                requires_confirmation=True
            ),
            FeatureFlag(
                id="emergency_stop",
                name="Emergency Stop",
                description="Allow emergency stop of all trading activities",
                category=FlagCategory.RISK,
                enabled=True,
                default=True
            ),
            FeatureFlag(
                id="experimental_strategies",
                name="Experimental Strategies",
                description="Enable experimental trading strategies",
                category=FlagCategory.EXPERIMENTAL,
                enabled=False,
                default=False,
                rollback_after_seconds=3600  # Auto-disable after 1 hour
            )
        ]
        
        for flag in default_flags:
            self.flags[flag.id] = flag
        
        # Save default flags
        self.save()
        logger.info(f"Created {len(default_flags)} default feature flags")
    
    def save(self) -> bool:
        """Save feature flags to storage.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        file_path = os.path.join(self.storage_dir, self.flags_file)
        try:
            # Create backup of existing file
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                try:
                    os.replace(file_path, backup_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup of feature flags: {e}")
            
            # Save new file
            with open(file_path, 'w') as f:
                data = {
                    "last_updated": datetime.now().isoformat(),
                    "flags": [flag.to_dict() for flag in self.flags.values()]
                }
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.flags)} feature flags to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
            return False
    
    def register_callback(self, callback: FlagChangeCallback):
        """Register a callback to be called when a flag changes.
        
        Args:
            callback: Function to call when a flag changes
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: FlagChangeCallback):
        """Unregister a previously registered callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, event: FlagChangeEvent):
        """Notify all registered callbacks of a flag change.
        
        Args:
            event: The flag change event
        """
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in feature flag callback: {e}")
    
    def create_flag(
        self,
        id: str,
        name: str,
        description: str,
        category: FlagCategory,
        default: bool = False,
        requires_confirmation: bool = False,
        rollback_after_seconds: Optional[int] = None,
        dependent_flags: Optional[Set[str]] = None,
        # New parameters
        rollout_percentage: int = 100,
        applicable_asset_classes: Optional[Set[str]] = None,
        context_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """Create a new feature flag.
        
        Args:
            id: Unique identifier for the flag
            name: Human-readable name for the flag
            description: Description of what the flag does
            category: Category of the flag
            default: Default state of the flag
            requires_confirmation: Whether changing requires confirmation
            rollback_after_seconds: Automatically disable after this many seconds
            dependent_flags: IDs of flags that depend on this flag
            rollout_percentage: Percentage of checks that should return enabled (0-100)
            applicable_asset_classes: Set of asset classes this flag applies to
            context_rules: List of rules for contextual evaluation
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        if id in self.flags:
            return False, f"Flag with ID '{id}' already exists"
        
        # Validate rollout percentage
        if not 0 <= rollout_percentage <= 100:
            return False, f"Rollout percentage must be between 0 and 100, got {rollout_percentage}"
        
        # Create context rules if provided
        rule_objects = []
        if context_rules:
            for rule_data in context_rules:
                try:
                    rule = ContextRule(
                        rule_type=rule_data["rule_type"],
                        parameters=rule_data["parameters"]
                    )
                    rule_objects.append(rule)
                except KeyError as e:
                    return False, f"Invalid context rule: missing {e}"
        
        flag = FeatureFlag(
            id=id,
            name=name,
            description=description,
            category=category,
            enabled=default,
            default=default,
            requires_confirmation=requires_confirmation,
            rollback_after_seconds=rollback_after_seconds,
            dependent_flags=dependent_flags or set(),
            # New fields
            rollout_percentage=rollout_percentage,
            applicable_asset_classes=applicable_asset_classes or {"ALL"},
            context_rules=rule_objects
        )
        
        # Add initial history entry
        flag.history.append(FlagChangeEvent(
            flag_id=id,
            enabled=default,
            changed_by="system",
            reason="Flag created"
        ))
        
        self.flags[id] = flag
        
        # Set up rollback timer if needed
        if default and rollback_after_seconds:
            self._setup_rollback_timer(id)
        
        if self.auto_save:
            self.save()
            
        logger.info(f"Created new feature flag: {id} (enabled={default})")
        return True, f"Flag '{id}' created successfully"
    
    def _setup_rollback_timer(self, flag_id: str):
        """Set up a timer to automatically disable a flag after a set time.
        
        Args:
            flag_id: ID of the flag to set up rollback for
        """
        flag = self.flags.get(flag_id)
        if not flag or not flag.rollback_after_seconds:
            return
        
        # Cancel existing timer if there is one
        if flag_id in self._rollback_timers:
            self._rollback_timers[flag_id].cancel()
            del self._rollback_timers[flag_id]
        
        # Set up new timer
        def rollback():
            self.set_flag(
                flag_id=flag_id, 
                enabled=False, 
                changed_by="system", 
                reason="Automatic rollback"
            )
            if flag_id in self._rollback_timers:
                del self._rollback_timers[flag_id]
        
        timer = threading.Timer(flag.rollback_after_seconds, rollback)
        timer.daemon = True
        timer.name = f"FlagRollback-{flag_id}"
        timer.start()
        
        self._rollback_timers[flag_id] = timer
        logger.info(f"Set up rollback timer for flag '{flag_id}' ({flag.rollback_after_seconds}s)")
    
    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Get a feature flag by ID.
        
        Args:
            flag_id: ID of the flag to get
            
        Returns:
            Optional[FeatureFlag]: The flag if found, None otherwise
        """
        return self.flags.get(flag_id)
    
    def is_enabled(self, flag_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled in the given context.
        
        Args:
            flag_id: ID of the flag to check
            context: Optional context for contextual evaluation
            
        Returns:
            bool: True if the flag is enabled, False otherwise
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False
        
        # If flag is disabled, no need for further checks
        if not flag.enabled:
            return False
        
        # Use empty context if none provided
        context = context or {}
        
        # Apply percentage rollout using consistent hashing
        if flag.rollout_percentage < 100:
            # Generate a hash based on flag ID and any stable identifiers in context
            # This ensures the same context always gets the same result
            hash_input = flag_id
            
            # Add any stable identifier that shouldn't change between checks
            if "user_id" in context:
                hash_input += str(context["user_id"])
            elif "account_id" in context:
                hash_input += str(context["account_id"])
            elif "symbol" in context:
                hash_input += str(context["symbol"])
            
            # Generate a hash and convert to a percentage (0-100)
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100
            
            # If hash value is greater than rollout percentage, disable
            if hash_value >= flag.rollout_percentage:
                return False
        
        # Check asset class if specified
        if "asset_class" in context and "ALL" not in flag.applicable_asset_classes:
            asset_class = context["asset_class"]
            if asset_class not in flag.applicable_asset_classes:
                return False
        
        # Apply context rules if any exist
        for rule in flag.context_rules:
            if not rule.evaluate(context):
                return False
        
        return True
    
    def list_flags(self, category: Optional[FlagCategory] = None) -> List[FeatureFlag]:
        """List all feature flags, optionally filtered by category.
        
        Args:
            category: Category to filter by, or None for all flags
            
        Returns:
            List[FeatureFlag]: List of flags
        """
        if category is None:
            return list(self.flags.values())
        
        return [flag for flag in self.flags.values() if flag.category == category]
    
    def set_flag(
        self, 
        flag_id: str, 
        enabled: bool, 
        changed_by: str = "system",
        reason: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Set the state of a feature flag.
        
        Args:
            flag_id: ID of the flag to set
            enabled: New state of the flag
            changed_by: Identifier of who changed the flag
            reason: Reason for the change
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        # No change needed
        if flag.enabled == enabled:
            return True, f"Flag '{flag_id}' already {'enabled' if enabled else 'disabled'}"
        
        # Update flag
        flag.enabled = enabled
        flag.modified_at = datetime.now()
        
        # Record change in history
        event = FlagChangeEvent(
            flag_id=flag_id,
            enabled=enabled,
            changed_by=changed_by,
            reason=reason
        )
        flag.history.append(event)
        
        # Handle dependent flags
        if not enabled:
            # When disabling a flag, check if any other flags depend on it
            for dep_flag in self.flags.values():
                if flag_id in dep_flag.dependent_flags and dep_flag.enabled:
                    # Disable dependent flag
                    dep_event = FlagChangeEvent(
                        flag_id=dep_flag.id,
                        enabled=False,
                        changed_by="system",
                        reason=f"Dependency '{flag_id}' was disabled"
                    )
                    dep_flag.enabled = False
                    dep_flag.modified_at = datetime.now()
                    dep_flag.history.append(dep_event)
                    self._notify_callbacks(dep_event)
                    logger.info(f"Auto-disabled dependent flag '{dep_flag.id}'")
        
        # Set up or cancel rollback timer
        if enabled and flag.rollback_after_seconds:
            self._setup_rollback_timer(flag_id)
        elif not enabled and flag_id in self._rollback_timers:
            self._rollback_timers[flag_id].cancel()
            del self._rollback_timers[flag_id]
        
        # Notify callbacks
        self._notify_callbacks(event)
        
        if self.auto_save:
            self.save()
        
        logger.info(f"Set feature flag '{flag_id}' to {'enabled' if enabled else 'disabled'} by {changed_by}")
        return True, f"Flag '{flag_id}' {'enabled' if enabled else 'disabled'} successfully"
    
    def delete_flag(self, flag_id: str) -> Tuple[bool, str]:
        """Delete a feature flag.
        
        Args:
            flag_id: ID of the flag to delete
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        if flag_id not in self.flags:
            return False, f"Flag with ID '{flag_id}' not found"
        
        # Check if any other flags depend on this one
        dependent_flags = []
        for dep_flag in self.flags.values():
            if flag_id in dep_flag.dependent_flags:
                dependent_flags.append(dep_flag.id)
        
        if dependent_flags:
            return False, f"Cannot delete flag '{flag_id}' because flags depend on it: {', '.join(dependent_flags)}"
        
        # Clean up rollback timer if exists
        if flag_id in self._rollback_timers:
            self._rollback_timers[flag_id].cancel()
            del self._rollback_timers[flag_id]
        
        # Delete flag
        del self.flags[flag_id]
        
        if self.auto_save:
            self.save()
        
        logger.info(f"Deleted feature flag '{flag_id}'")
        return True, f"Flag '{flag_id}' deleted successfully"
    
    def get_flag_history(self, flag_id: str) -> List[FlagChangeEvent]:
        """Get the history of changes for a flag.
        
        Args:
            flag_id: ID of the flag to get history for
            
        Returns:
            List[FlagChangeEvent]: List of change events
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return []
        
        return flag.history
    
    def reset_flag(self, flag_id: str) -> Tuple[bool, str]:
        """Reset a flag to its default state.
        
        Args:
            flag_id: ID of the flag to reset
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        return self.set_flag(
            flag_id=flag_id,
            enabled=flag.default,
            changed_by="system",
            reason="Reset to default"
        )
    
    def reset_all_flags(self) -> Tuple[int, int]:
        """Reset all flags to their default states.
        
        Returns:
            Tuple[int, int]: (Number of flags reset, Number of errors)
        """
        success_count = 0
        error_count = 0
        
        for flag_id in self.flags:
            success, _ = self.reset_flag(flag_id)
            if success:
                success_count += 1
            else:
                error_count += 1
        
        return success_count, error_count
    
    def cleanup(self):
        """Clean up resources used by the service."""
        # Stop auto-save thread
        if self._save_thread and self._save_thread.is_alive():
            self._save_stop_event.set()
            self._save_thread.join(timeout=5)
        
        # Cancel all rollback timers
        for timer in self._rollback_timers.values():
            timer.cancel()
        self._rollback_timers.clear()
        
        # Save flags one last time
        self.save()

    def update_flag_rollout(self, flag_id: str, rollout_percentage: int) -> Tuple[bool, str]:
        """Update the rollout percentage for a flag.
        
        Args:
            flag_id: ID of the flag to update
            rollout_percentage: New rollout percentage (0-100)
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        if not 0 <= rollout_percentage <= 100:
            return False, f"Rollout percentage must be between 0 and 100, got {rollout_percentage}"
        
        flag.rollout_percentage = rollout_percentage
        flag.modified_at = datetime.now()
        
        # Add to history
        event = FlagChangeEvent(
            flag_id=flag_id,
            enabled=flag.enabled,  # No change to enabled state
            changed_by="system",
            reason=f"Updated rollout to {rollout_percentage}%"
        )
        flag.history.append(event)
        
        if self.auto_save:
            self.save()
        
        return True, f"Updated rollout percentage for '{flag_id}' to {rollout_percentage}%"
    
    def update_flag_asset_classes(self, flag_id: str, asset_classes: Set[str]) -> Tuple[bool, str]:
        """Update the applicable asset classes for a flag.
        
        Args:
            flag_id: ID of the flag to update
            asset_classes: New set of applicable asset classes
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        # Validate asset classes
        try:
            for ac in asset_classes:
                if ac != "ALL" and not hasattr(AssetClass, ac):
                    return False, f"Invalid asset class: {ac}"
        except TypeError:
            return False, "Asset classes must be a set of strings"
        
        flag.applicable_asset_classes = asset_classes
        flag.modified_at = datetime.now()
        
        # Add to history
        event = FlagChangeEvent(
            flag_id=flag_id,
            enabled=flag.enabled,  # No change to enabled state
            changed_by="system",
            reason=f"Updated asset classes to {', '.join(asset_classes)}"
        )
        flag.history.append(event)
        
        if self.auto_save:
            self.save()
        
        return True, f"Updated asset classes for '{flag_id}'"
    
    def add_context_rule(self, flag_id: str, rule_type: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Add a context rule to a flag.
        
        Args:
            flag_id: ID of the flag to update
            rule_type: Type of rule to add
            parameters: Parameters for the rule
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        # Validate rule type
        valid_rule_types = ["asset_class", "time_window", "account_value", "market_condition"]
        if rule_type not in valid_rule_types:
            return False, f"Invalid rule type: {rule_type}. Valid types: {', '.join(valid_rule_types)}"
        
        # Create and add the rule
        rule = ContextRule(rule_type=rule_type, parameters=parameters)
        flag.context_rules.append(rule)
        flag.modified_at = datetime.now()
        
        # Add to history
        event = FlagChangeEvent(
            flag_id=flag_id,
            enabled=flag.enabled,  # No change to enabled state
            changed_by="system",
            reason=f"Added {rule_type} rule"
        )
        flag.history.append(event)
        
        if self.auto_save:
            self.save()
        
        return True, f"Added {rule_type} rule to '{flag_id}'"
    
    def remove_context_rule(self, flag_id: str, rule_index: int) -> Tuple[bool, str]:
        """Remove a context rule from a flag.
        
        Args:
            flag_id: ID of the flag to update
            rule_index: Index of the rule to remove
            
        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return False, f"Flag with ID '{flag_id}' not found"
        
        if not 0 <= rule_index < len(flag.context_rules):
            return False, f"Invalid rule index: {rule_index}"
        
        # Remove the rule
        removed_rule = flag.context_rules.pop(rule_index)
        flag.modified_at = datetime.now()
        
        # Add to history
        event = FlagChangeEvent(
            flag_id=flag_id,
            enabled=flag.enabled,  # No change to enabled state
            changed_by="system",
            reason=f"Removed {removed_rule.rule_type} rule"
        )
        flag.history.append(event)
        
        if self.auto_save:
            self.save()
        
        return True, f"Removed rule from '{flag_id}'"


def get_feature_flag_service() -> FeatureFlagService:
    """Get the singleton instance of the feature flag service.
    
    Returns:
        FeatureFlagService: The feature flag service instance
    """
    with FeatureFlagService._instance_lock:
        if FeatureFlagService._instance is None:
            FeatureFlagService._instance = FeatureFlagService()
        return FeatureFlagService._instance 