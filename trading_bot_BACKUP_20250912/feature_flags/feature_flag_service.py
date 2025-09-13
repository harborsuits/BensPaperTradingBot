#!/usr/bin/env python3
"""
Feature Flag Service for Trading Bot

This module provides a centralized service for managing feature flags that can be used
to enable/disable specific features or strategies in the trading bot without requiring
full deployment. Useful for A/B testing, gradual rollouts, and emergency controls.
"""

import os
import json
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from threading import Lock

logger = logging.getLogger(__name__)

class FeatureFlagCategory(str, Enum):
    """Categories for organizing feature flags"""
    STRATEGY = "strategy"
    RISK = "risk"
    DATA = "data"
    EXECUTION = "execution"
    NOTIFICATION = "notification"
    SYSTEM = "system"
    OTHER = "other"

class FeatureFlag:
    """
    Represents a feature flag with its properties and state
    """
    def __init__(
        self, 
        name: str, 
        enabled: bool = False,
        category: FeatureFlagCategory = FeatureFlagCategory.OTHER,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.enabled = enabled
        self.category = category
        self.description = description
        self.metadata = metadata or {}
        self.last_updated = time.time()
        self.last_modified_by = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the feature flag to dictionary for serialization"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "category": self.category.value,
            "description": self.description,
            "metadata": self.metadata,
            "last_updated": self.last_updated,
            "last_modified_by": self.last_modified_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureFlag':
        """Create a feature flag from dictionary"""
        flag = cls(
            name=data["name"],
            enabled=data["enabled"],
            category=FeatureFlagCategory(data["category"]),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
        flag.last_updated = data.get("last_updated", time.time())
        flag.last_modified_by = data.get("last_modified_by", "system")
        return flag


class FeatureFlagService:
    """
    Service for managing feature flags with persistence.
    Handles flag creation, retrieval, and state changes.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FeatureFlagService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, storage_path: str = None):
        """Initialize the feature flag service"""
        if self._initialized:
            return
            
        self._initialized = True
        self._flags: Dict[str, FeatureFlag] = {}
        self._listeners = set()
        self._lock = Lock()
        
        # Set storage path with default fallback
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", 
            "feature_flags.json"
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Load existing flags
        self._load_flags()
        
        # Add default flags if none exist
        if not self._flags:
            self._add_default_flags()
            self._save_flags()
            
        logger.info(f"Feature Flag Service initialized with {len(self._flags)} flags")
    
    def _add_default_flags(self):
        """Add default feature flags on first initialization"""
        defaults = [
            FeatureFlag("enable_trading", True, FeatureFlagCategory.SYSTEM, 
                       "Master switch to enable/disable all trading activity"),
            FeatureFlag("enable_risk_limits", True, FeatureFlagCategory.RISK,
                       "Enable risk management limits"),
            FeatureFlag("enable_notifications", True, FeatureFlagCategory.NOTIFICATION,
                       "Enable sending of notifications"),
            FeatureFlag("debug_mode", False, FeatureFlagCategory.SYSTEM,
                       "Enable additional debug logging and checks"),
            FeatureFlag("maintenance_mode", False, FeatureFlagCategory.SYSTEM,
                       "Put system in maintenance mode (read-only)"),
        ]
        
        for flag in defaults:
            self._flags[flag.name] = flag
    
    def _load_flags(self):
        """Load feature flags from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for flag_data in data:
                        flag = FeatureFlag.from_dict(flag_data)
                        self._flags[flag.name] = flag
                logger.info(f"Loaded {len(self._flags)} feature flags from {self.storage_path}")
            else:
                logger.info(f"No feature flags file found at {self.storage_path}")
        except Exception as e:
            logger.error(f"Error loading feature flags: {e}")
    
    def _save_flags(self):
        """Save feature flags to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump([flag.to_dict() for flag in self._flags.values()], f, indent=2)
            logger.info(f"Saved {len(self._flags)} feature flags to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving feature flags: {e}")
    
    def add_flag(self, flag: FeatureFlag, modified_by: str = "system") -> bool:
        """Add a new feature flag"""
        with self._lock:
            if flag.name in self._flags:
                logger.warning(f"Feature flag '{flag.name}' already exists, use update instead")
                return False
            
            flag.last_modified_by = modified_by
            flag.last_updated = time.time()
            self._flags[flag.name] = flag
            self._save_flags()
            self._notify_listeners(flag.name, flag.enabled)
            logger.info(f"Added feature flag '{flag.name}' (enabled: {flag.enabled}) by {modified_by}")
            return True
    
    def update_flag(self, name: str, enabled: bool, modified_by: str = "system") -> bool:
        """Update the state of a feature flag"""
        with self._lock:
            if name not in self._flags:
                logger.warning(f"Feature flag '{name}' does not exist")
                return False
            
            if self._flags[name].enabled == enabled:
                logger.debug(f"Feature flag '{name}' already set to {enabled}")
                return True
            
            self._flags[name].enabled = enabled
            self._flags[name].last_updated = time.time()
            self._flags[name].last_modified_by = modified_by
            self._save_flags()
            self._notify_listeners(name, enabled)
            logger.info(f"Updated feature flag '{name}' to {enabled} by {modified_by}")
            return True
    
    def delete_flag(self, name: str, modified_by: str = "system") -> bool:
        """Delete a feature flag"""
        with self._lock:
            if name not in self._flags:
                logger.warning(f"Feature flag '{name}' does not exist")
                return False
            
            flag = self._flags.pop(name)
            self._save_flags()
            self._notify_listeners(name, None)  # None indicates deletion
            logger.info(f"Deleted feature flag '{name}' by {modified_by}")
            return True
    
    def is_enabled(self, name: str) -> bool:
        """Check if a feature flag is enabled"""
        with self._lock:
            if name not in self._flags:
                logger.warning(f"Feature flag '{name}' does not exist, defaulting to False")
                return False
            
            return self._flags[name].enabled
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name"""
        with self._lock:
            return self._flags.get(name)
    
    def get_all_flags(self) -> List[FeatureFlag]:
        """Get all feature flags"""
        with self._lock:
            return list(self._flags.values())
    
    def get_flags_by_category(self, category: FeatureFlagCategory) -> List[FeatureFlag]:
        """Get feature flags by category"""
        with self._lock:
            return [flag for flag in self._flags.values() if flag.category == category]
    
    def register_listener(self, callback):
        """Register a listener to be notified on flag changes"""
        self._listeners.add(callback)
        return callback  # Return callback for easy unregistering
    
    def unregister_listener(self, callback):
        """Unregister a listener"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify_listeners(self, flag_name: str, state: Optional[bool]):
        """Notify listeners of flag changes"""
        for listener in self._listeners:
            try:
                listener(flag_name, state)
            except Exception as e:
                logger.error(f"Error notifying listener for flag '{flag_name}': {e}")


# Convenience function to get the singleton instance
def get_feature_flag_service(storage_path: str = None) -> FeatureFlagService:
    """Get the singleton instance of the feature flag service"""
    return FeatureFlagService(storage_path) 