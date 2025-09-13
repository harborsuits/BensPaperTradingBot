#!/usr/bin/env python3
"""
Broker Intelligence Configuration Manager

Provides hot-reload capabilities for broker intelligence configurations,
enabling runtime adjustments of thresholds, weights, and parameters
without requiring system restarts.

Also supports A/B testing of different configuration profiles to
optimize broker selection and circuit breaker behavior.
"""

import os
import json
import time
import random
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

from trading_bot.event_system.event_bus import EventBus
from trading_bot.event_system.event_types import EventType


logger = logging.getLogger(__name__)


class ConfigChangeEvent:
    """Event emitted when configuration changes"""
    
    def __init__(self, config_type: str, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        self.config_type = config_type
        self.old_config = old_config
        self.new_config = new_config
        self.timestamp = datetime.now()
    
    def get_changes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of changes between old and new configs
        
        Returns:
            Dict with changed_values, added_values, and removed_values
        """
        changes = {
            "changed_values": {},
            "added_values": {},
            "removed_values": {}
        }
        
        # Find changed and removed values
        for key, old_value in self.old_config.items():
            if key in self.new_config:
                # If value is a nested dict, compare recursively
                if isinstance(old_value, dict) and isinstance(self.new_config[key], dict):
                    nested_changes = self._compare_dicts(old_value, self.new_config[key])
                    if nested_changes:
                        changes["changed_values"][key] = nested_changes
                # Otherwise compare values directly
                elif self.new_config[key] != old_value:
                    changes["changed_values"][key] = {
                        "old": old_value,
                        "new": self.new_config[key]
                    }
            else:
                changes["removed_values"][key] = old_value
        
        # Find added values
        for key, new_value in self.new_config.items():
            if key not in self.old_config:
                changes["added_values"][key] = new_value
        
        return changes
    
    def _compare_dicts(self, old_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two dictionaries and return differences"""
        result = {}
        
        # Find changed values
        for key, old_value in old_dict.items():
            if key in new_dict:
                # Recursive comparison for nested dicts
                if isinstance(old_value, dict) and isinstance(new_dict[key], dict):
                    nested_diff = self._compare_dicts(old_value, new_dict[key])
                    if nested_diff:
                        result[key] = nested_diff
                # Direct comparison for other values
                elif new_dict[key] != old_value:
                    result[key] = {
                        "old": old_value,
                        "new": new_dict[key]
                    }
        
        # Find added/removed keys (just note that they changed)
        added_removed = set(new_dict.keys()) ^ set(old_dict.keys())
        for key in added_removed:
            if key in new_dict:
                result[key] = {"added": new_dict[key]}
            else:
                result[key] = {"removed": old_dict[key]}
        
        return result


class ABTestProfile:
    """Configuration profile for A/B testing"""
    
    def __init__(
        self,
        profile_id: str,
        config: Dict[str, Any], 
        weight: float = 1.0,
        description: str = "",
        active: bool = True
    ):
        """
        Initialize A/B test profile
        
        Args:
            profile_id: Unique identifier for this profile
            config: Configuration data for this profile
            weight: Selection weight (higher = more likely to be chosen)
            description: Human-readable description of this profile
            active: Whether this profile is active and available for selection
        """
        self.profile_id = profile_id
        self.config = config
        self.weight = weight
        self.description = description
        self.active = active
        
        # Tracking metrics
        self.times_selected = 0
        self.last_selected = None
        self.performance_metrics = {}
    
    def mark_selected(self):
        """Mark this profile as having been selected"""
        self.times_selected += 1
        self.last_selected = datetime.now()
    
    def update_metric(self, metric_name: str, value: float):
        """
        Update performance metric for this profile
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = {
                "values": [],
                "total": 0,
                "count": 0,
                "min": value,
                "max": value
            }
        
        metric = self.performance_metrics[metric_name]
        metric["values"].append(value)
        metric["total"] += value
        metric["count"] += 1
        metric["min"] = min(metric["min"], value)
        metric["max"] = max(metric["max"], value)
        
        # Keep only the last 100 values to limit memory usage
        if len(metric["values"]) > 100:
            oldest = metric["values"].pop(0)
            metric["total"] -= oldest
    
    def get_metric_average(self, metric_name: str) -> float:
        """Get average value for a performance metric"""
        if metric_name not in self.performance_metrics:
            return 0
        
        metric = self.performance_metrics[metric_name]
        if metric["count"] == 0:
            return 0
        
        return metric["total"] / metric["count"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "profile_id": self.profile_id,
            "config": self.config,
            "weight": self.weight,
            "description": self.description,
            "active": self.active,
            "times_selected": self.times_selected,
            "last_selected": self.last_selected.isoformat() if self.last_selected else None,
            "metrics_summary": {
                name: {
                    "avg": self.get_metric_average(name),
                    "min": metric["min"],
                    "max": metric["max"],
                    "count": metric["count"]
                }
                for name, metric in self.performance_metrics.items()
            }
        }


class BrokerIntelligenceConfigManager:
    """
    Manager for broker intelligence configuration
    
    Supports hot-reloading of configuration files and A/B testing
    of different configuration profiles.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config_path: str,
        refresh_interval: int = 60,
        ab_test_enabled: bool = False
    ):
        """
        Initialize the configuration manager
        
        Args:
            event_bus: Event bus for emitting config change events
            config_path: Path to configuration file
            refresh_interval: Seconds between config refresh checks
            ab_test_enabled: Whether A/B testing is enabled
        """
        self.event_bus = event_bus
        self.config_path = config_path
        self.refresh_interval = refresh_interval
        self.ab_test_enabled = ab_test_enabled
        
        # Initialize file state
        self.config_last_modified = 0
        self.config_file_size = 0
        
        # Load initial configuration
        self.current_config = self._load_config()
        self._update_file_state()
        
        # A/B test profiles
        self.ab_test_profiles = {}
        self.current_ab_profile = None
        
        # Load A/B test profiles if enabled
        if ab_test_enabled:
            self._load_ab_test_profiles()
        
        # Start background refresh thread
        self.refresh_thread = None
        self.stop_refresh = False
        self._start_refresh_thread()
        
        logger.info(f"BrokerIntelligenceConfigManager initialized with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            # Return empty config
            return {}
    
    def _update_file_state(self):
        """Update file state for change detection"""
        try:
            stats = os.stat(self.config_path)
            self.config_last_modified = stats.st_mtime
            self.config_file_size = stats.st_size
        except Exception as e:
            logger.error(f"Failed to stat config file: {str(e)}")
    
    def _check_config_changed(self) -> bool:
        """Check if config file has changed since last load"""
        try:
            stats = os.stat(self.config_path)
            return (
                stats.st_mtime > self.config_last_modified or
                stats.st_size != self.config_file_size
            )
        except Exception:
            return False
    
    def _start_refresh_thread(self):
        """Start background thread for config refresh"""
        if self.refresh_thread is not None:
            return
        
        self.stop_refresh = False
        self.refresh_thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name="ConfigRefreshThread"
        )
        self.refresh_thread.start()
        
        logger.info("Started configuration refresh thread")
    
    def _refresh_loop(self):
        """Background thread loop for config refresh"""
        while not self.stop_refresh:
            try:
                # Check if config file has changed
                if self._check_config_changed():
                    self.refresh_config()
                
                # Sleep for refresh interval
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in config refresh loop: {str(e)}")
                # Sleep briefly to avoid tight loop on persistent errors
                time.sleep(5)
    
    def refresh_config(self) -> bool:
        """
        Reload configuration from file if changed
        
        Returns:
            bool: True if config was refreshed, False otherwise
        """
        try:
            # Load new config
            new_config = self._load_config()
            
            # Check if config actually changed
            if new_config == self.current_config:
                # Update file state even if content didn't change
                self._update_file_state()
                return False
            
            # Config changed, update and emit event
            old_config = self.current_config
            self.current_config = new_config
            self._update_file_state()
            
            # Emit config change event
            event = ConfigChangeEvent("broker_intelligence", old_config, new_config)
            self.event_bus.emit(
                event_type=EventType.CONFIG_CHANGED,
                data={
                    "event_subtype": "broker_intelligence_config",
                    "config_type": "broker_intelligence",
                    "timestamp": datetime.now().isoformat(),
                    "changes": event.get_changes()
                }
            )
            
            logger.info("Broker intelligence configuration refreshed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh config: {str(e)}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        If A/B testing is enabled, returns the currently selected profile's config.
        Otherwise, returns the main config.
        
        Returns:
            Dict: Current configuration
        """
        if self.ab_test_enabled and self.current_ab_profile:
            return self.current_ab_profile.config
        return self.current_config
    
    def _load_ab_test_profiles(self):
        """Load A/B test profiles from config"""
        profiles_config = self.current_config.get("ab_test_profiles", {})
        
        for profile_id, profile_data in profiles_config.items():
            if isinstance(profile_data, dict) and profile_data.get("config"):
                self.ab_test_profiles[profile_id] = ABTestProfile(
                    profile_id=profile_id,
                    config=profile_data["config"],
                    weight=profile_data.get("weight", 1.0),
                    description=profile_data.get("description", ""),
                    active=profile_data.get("active", True)
                )
        
        logger.info(f"Loaded {len(self.ab_test_profiles)} A/B test profiles")
    
    def select_ab_test_profile(self, force_profile_id: Optional[str] = None) -> Optional[str]:
        """
        Select an A/B test profile based on weights
        
        Args:
            force_profile_id: If provided, forces selection of this profile
            
        Returns:
            str: Selected profile ID or None if no profiles available
        """
        if not self.ab_test_enabled or not self.ab_test_profiles:
            return None
        
        # Force selection of specific profile if requested
        if force_profile_id and force_profile_id in self.ab_test_profiles:
            profile = self.ab_test_profiles[force_profile_id]
            if profile.active:
                self.current_ab_profile = profile
                profile.mark_selected()
                return profile.profile_id
        
        # Get active profiles with their weights
        active_profiles = {
            pid: profile.weight 
            for pid, profile in self.ab_test_profiles.items() 
            if profile.active
        }
        
        if not active_profiles:
            logger.warning("No active A/B test profiles available")
            return None
        
        # Select profile based on weights
        total_weight = sum(active_profiles.values())
        selection = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for profile_id, weight in active_profiles.items():
            cumulative_weight += weight
            if selection <= cumulative_weight:
                # Found our selection
                profile = self.ab_test_profiles[profile_id]
                self.current_ab_profile = profile
                profile.mark_selected()
                return profile_id
        
        # Fallback in case of rounding errors
        pid = list(active_profiles.keys())[0]
        self.current_ab_profile = self.ab_test_profiles[pid]
        self.current_ab_profile.mark_selected()
        return pid
    
    def record_ab_test_metric(self, metric_name: str, value: float):
        """
        Record a performance metric for the current A/B test profile
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if not self.ab_test_enabled or not self.current_ab_profile:
            return
        
        self.current_ab_profile.update_metric(metric_name, value)
    
    def get_ab_test_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all A/B test profiles
        
        Returns:
            Dict mapping profile IDs to their metrics
        """
        if not self.ab_test_enabled:
            return {}
        
        return {
            profile_id: profile.to_dict()
            for profile_id, profile in self.ab_test_profiles.items()
        }
    
    def add_ab_test_profile(
        self, 
        profile_id: str,
        config: Dict[str, Any],
        weight: float = 1.0,
        description: str = "",
        active: bool = True
    ) -> bool:
        """
        Add a new A/B test profile
        
        Args:
            profile_id: Unique identifier for the profile
            config: Configuration for this profile
            weight: Selection weight
            description: Human-readable description
            active: Whether the profile is active
            
        Returns:
            bool: True if profile was added, False if already exists
        """
        if not self.ab_test_enabled:
            logger.warning("A/B testing is disabled")
            return False
        
        if profile_id in self.ab_test_profiles:
            logger.warning(f"A/B test profile '{profile_id}' already exists")
            return False
        
        # Add new profile
        self.ab_test_profiles[profile_id] = ABTestProfile(
            profile_id=profile_id,
            config=config,
            weight=weight,
            description=description,
            active=active
        )
        
        logger.info(f"Added A/B test profile: {profile_id}")
        return True
    
    def update_ab_test_profile(
        self,
        profile_id: str,
        config: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = None,
        description: Optional[str] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        Update an existing A/B test profile
        
        Args:
            profile_id: ID of profile to update
            config: New configuration (if None, unchanged)
            weight: New weight (if None, unchanged)
            description: New description (if None, unchanged)
            active: New active status (if None, unchanged)
            
        Returns:
            bool: True if profile was updated, False if not found
        """
        if not self.ab_test_enabled:
            return False
        
        if profile_id not in self.ab_test_profiles:
            logger.warning(f"A/B test profile '{profile_id}' not found")
            return False
        
        profile = self.ab_test_profiles[profile_id]
        
        # Update fields if provided
        if config is not None:
            profile.config = config
        
        if weight is not None:
            profile.weight = weight
        
        if description is not None:
            profile.description = description
        
        if active is not None:
            profile.active = active
        
        logger.info(f"Updated A/B test profile: {profile_id}")
        return True
    
    def remove_ab_test_profile(self, profile_id: str) -> bool:
        """
        Remove an A/B test profile
        
        Args:
            profile_id: ID of profile to remove
            
        Returns:
            bool: True if profile was removed, False if not found
        """
        if not self.ab_test_enabled:
            return False
        
        if profile_id not in self.ab_test_profiles:
            return False
        
        # Remove profile
        del self.ab_test_profiles[profile_id]
        
        # Reset current profile if it was the removed one
        if self.current_ab_profile and self.current_ab_profile.profile_id == profile_id:
            self.current_ab_profile = None
        
        logger.info(f"Removed A/B test profile: {profile_id}")
        return True
    
    def save_ab_test_profiles(self) -> bool:
        """
        Save A/B test profiles to config file
        
        Returns:
            bool: True if profiles were saved, False on error
        """
        if not self.ab_test_enabled:
            return False
        
        try:
            # Load current config
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Update A/B test profiles
            config["ab_test_profiles"] = {
                profile_id: {
                    "config": profile.config,
                    "weight": profile.weight,
                    "description": profile.description,
                    "active": profile.active
                }
                for profile_id, profile in self.ab_test_profiles.items()
            }
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update file state
            self._update_file_state()
            
            logger.info(f"Saved {len(self.ab_test_profiles)} A/B test profiles to config")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save A/B test profiles: {str(e)}")
            return False
    
    def shutdown(self):
        """Shutdown the config manager and stop refresh thread"""
        self.stop_refresh = True
        if self.refresh_thread:
            self.refresh_thread.join(timeout=1.0)
            self.refresh_thread = None
        
        logger.info("Broker intelligence config manager shut down")
