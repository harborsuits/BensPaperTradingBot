#!/usr/bin/env python3
"""
Feature Flag Service

This module provides a feature flag system for dynamically enabling/disabling 
features in the trading bot without requiring redeployment.

Features can be toggled via API, CLI, or Telegram commands.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime, timedelta
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_flags")

class FeatureFlag:
    """Represents a feature flag with metadata."""
    
    def __init__(
        self,
        name: str,
        enabled: bool = False,
        description: str = "",
        category: str = "general",
        expiry: Optional[datetime] = None,
        created_by: str = "system",
        created_at: Optional[datetime] = None,
        updated_by: str = "system",
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a feature flag.
        
        Args:
            name: Unique identifier for the flag
            enabled: Whether the flag is enabled
            description: Description of the flag's purpose
            category: Category for grouping flags
            expiry: Optional expiry time for the flag
            created_by: Who created the flag
            created_at: When the flag was created
            updated_by: Who last updated the flag
            updated_at: When the flag was last updated
            metadata: Additional data associated with the flag
        """
        self.name = name
        self.enabled = enabled
        self.description = description
        self.category = category
        self.expiry = expiry
        self.created_by = created_by
        self.created_at = created_at or datetime.now()
        self.updated_by = updated_by
        self.updated_at = updated_at or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "category": self.category,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureFlag':
        """Create from dictionary representation."""
        expiry = data.get("expiry")
        if expiry:
            expiry = datetime.fromisoformat(expiry)
        
        return cls(
            name=data["name"],
            enabled=data["enabled"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            expiry=expiry,
            created_by=data.get("created_by", "system"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_by=data.get("updated_by", "system"),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )
    
    def is_expired(self) -> bool:
        """Check if the flag has expired."""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry


class FeatureFlagService:
    """
    Service for managing feature flags with Redis backend.
    
    This service allows for dynamic enabling/disabling of features
    and persists the state in Redis.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        prefix: str = "trading_bot:feature_flag:",
        default_ttl: int = 0, # 0 means no expiration
        auto_refresh: bool = True,
        refresh_interval: int = 60
    ):
        """
        Initialize the feature flag service.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database to use
            redis_password: Optional Redis password
            prefix: Prefix for Redis keys
            default_ttl: Default TTL for flags in seconds (0 for no expiration)
            auto_refresh: Whether to auto-refresh flags from storage
            refresh_interval: Interval for auto-refresh in seconds
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        self.prefix = prefix
        self.default_ttl = default_ttl
        
        # In-memory cache of flags
        self.flags: Dict[str, FeatureFlag] = {}
        self.last_refresh = datetime.now()
        
        # Listeners for flag changes
        self.listeners: Dict[str, List[callable]] = {}
        
        # For auto-refresh
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self.refresh_thread = None
        self.running = False
        
        # Load initial flags
        self.refresh_flags()
        
        # Start auto-refresh if enabled
        if self.auto_refresh:
            self.start_auto_refresh()
    
    def start_auto_refresh(self) -> None:
        """Start the auto-refresh thread."""
        if self.refresh_thread is not None and self.refresh_thread.is_alive():
            logger.warning("Auto-refresh thread already running")
            return
        
        self.running = True
        self.refresh_thread = threading.Thread(
            target=self._auto_refresh_loop,
            daemon=True,
            name="feature-flag-refresh"
        )
        self.refresh_thread.start()
        logger.info("Started feature flag auto-refresh")
    
    def stop_auto_refresh(self) -> None:
        """Stop the auto-refresh thread."""
        self.running = False
        if self.refresh_thread:
            self.refresh_thread.join(timeout=10)
            logger.info("Stopped feature flag auto-refresh")
    
    def _auto_refresh_loop(self) -> None:
        """Background thread for auto-refreshing flags."""
        while self.running:
            try:
                self.refresh_flags()
            except Exception as e:
                logger.error(f"Error refreshing feature flags: {str(e)}", exc_info=True)
            
            # Sleep until next refresh
            time.sleep(self.refresh_interval)
    
    def refresh_flags(self) -> None:
        """Refresh flags from Redis."""
        try:
            # Get all flag keys
            keys = self.redis_client.keys(f"{self.prefix}*")
            
            # Fetch all flags in one batch
            if keys:
                flag_data = self.redis_client.mget(keys)
                
                # Process each flag
                for key, data in zip(keys, flag_data):
                    if data:
                        flag_name = key.replace(self.prefix, "")
                        try:
                            flag_dict = json.loads(data)
                            flag = FeatureFlag.from_dict(flag_dict)
                            
                            # Check for expiry
                            if flag.is_expired():
                                logger.info(f"Flag {flag.name} has expired, disabling")
                                flag.enabled = False
                                self._save_flag(flag)
                            
                            self.flags[flag_name] = flag
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding flag data for {key}")
            
            self.last_refresh = datetime.now()
            logger.debug(f"Refreshed {len(keys)} feature flags")
            
        except redis.RedisError as e:
            logger.error(f"Redis error refreshing flags: {str(e)}")
    
    def _save_flag(self, flag: FeatureFlag) -> bool:
        """Save a flag to Redis."""
        try:
            key = f"{self.prefix}{flag.name}"
            flag_data = json.dumps(flag.to_dict())
            
            # Set with TTL if expiry is set
            if flag.expiry:
                ttl = int((flag.expiry - datetime.now()).total_seconds())
                if ttl > 0:
                    self.redis_client.setex(key, ttl, flag_data)
                else:
                    # Already expired
                    self.redis_client.delete(key)
                    return False
            else:
                # No expiry, use default TTL
                if self.default_ttl > 0:
                    self.redis_client.setex(key, self.default_ttl, flag_data)
                else:
                    self.redis_client.set(key, flag_data)
            
            return True
        except redis.RedisError as e:
            logger.error(f"Redis error saving flag {flag.name}: {str(e)}")
            return False
    
    def create_flag(
        self,
        name: str,
        enabled: bool = False,
        description: str = "",
        category: str = "general",
        expiry: Optional[datetime] = None,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new feature flag.
        
        Args:
            name: Unique identifier for the flag
            enabled: Whether the flag is enabled
            description: Description of the flag's purpose
            category: Category for grouping flags
            expiry: Optional expiry time for the flag
            created_by: Who created the flag
            metadata: Additional data associated with the flag
            
        Returns:
            True if the flag was created successfully
        """
        # Check if flag already exists
        if name in self.flags:
            logger.warning(f"Flag {name} already exists")
            return False
        
        # Create the flag
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            description=description,
            category=category,
            expiry=expiry,
            created_by=created_by,
            metadata=metadata
        )
        
        # Save to Redis
        if self._save_flag(flag):
            # Update local cache
            self.flags[name] = flag
            logger.info(f"Created flag {name} ({category}) - enabled: {enabled}")
            
            # Notify listeners
            self._notify_listeners(name, enabled, "created")
            return True
        
        return False
    
    def update_flag(
        self,
        name: str,
        enabled: Optional[bool] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        expiry: Optional[datetime] = None,
        updated_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing feature flag.
        
        Args:
            name: Name of the flag to update
            enabled: New enabled state (None to keep current)
            description: New description (None to keep current)
            category: New category (None to keep current)
            expiry: New expiry time (None to keep current)
            updated_by: Who is updating the flag
            metadata: New metadata (None to keep current)
            
        Returns:
            True if the flag was updated successfully
        """
        # Check if flag exists
        if name not in self.flags:
            logger.warning(f"Flag {name} does not exist")
            return False
        
        flag = self.flags[name]
        old_enabled = flag.enabled
        
        # Update fields
        if enabled is not None:
            flag.enabled = enabled
        if description is not None:
            flag.description = description
        if category is not None:
            flag.category = category
        if expiry is not None:
            flag.expiry = expiry
        if metadata is not None:
            flag.metadata = metadata
        
        flag.updated_by = updated_by
        flag.updated_at = datetime.now()
        
        # Save to Redis
        if self._save_flag(flag):
            # Update local cache
            self.flags[name] = flag
            
            # Log if enabled state changed
            if old_enabled != flag.enabled:
                logger.info(f"Flag {name} {'enabled' if flag.enabled else 'disabled'} by {updated_by}")
                
                # Notify listeners if enabled state changed
                self._notify_listeners(name, flag.enabled, "updated")
            else:
                logger.info(f"Updated flag {name} (no state change)")
            
            return True
        
        return False
    
    def delete_flag(self, name: str, deleted_by: str = "system") -> bool:
        """
        Delete a feature flag.
        
        Args:
            name: Name of the flag to delete
            deleted_by: Who is deleting the flag
            
        Returns:
            True if the flag was deleted successfully
        """
        # Check if flag exists
        if name not in self.flags:
            logger.warning(f"Flag {name} does not exist")
            return False
        
        # Get current state before deleting
        was_enabled = self.flags[name].enabled
        
        # Delete from Redis
        try:
            key = f"{self.prefix}{name}"
            self.redis_client.delete(key)
            
            # Remove from local cache
            del self.flags[name]
            
            logger.info(f"Deleted flag {name} by {deleted_by}")
            
            # Notify listeners
            self._notify_listeners(name, False, "deleted")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error deleting flag {name}: {str(e)}")
            return False
    
    def is_enabled(self, name: str, default: bool = False) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            name: Name of the flag to check
            default: Default value if flag doesn't exist
            
        Returns:
            True if the flag is enabled, default if not found
        """
        # Refresh if flag not in cache
        if name not in self.flags:
            # Try to refresh from Redis
            try:
                key = f"{self.prefix}{name}"
                data = self.redis_client.get(key)
                if data:
                    try:
                        flag_dict = json.loads(data)
                        flag = FeatureFlag.from_dict(flag_dict)
                        
                        # Check for expiry
                        if flag.is_expired():
                            logger.info(f"Flag {flag.name} has expired, disabling")
                            flag.enabled = False
                            self._save_flag(flag)
                        
                        self.flags[name] = flag
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding flag data for {key}")
                        return default
                else:
                    return default
            except redis.RedisError:
                return default
        
        # Get from cache
        flag = self.flags.get(name)
        if flag:
            # Check if expired
            if flag.is_expired():
                logger.info(f"Flag {flag.name} has expired, disabling")
                flag.enabled = False
                self._save_flag(flag)
                return False
            
            return flag.enabled
        
        return default
    
    def get_all_flags(self, category: Optional[str] = None) -> List[FeatureFlag]:
        """
        Get all feature flags.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of feature flags
        """
        # Ensure we have the latest flags
        self.refresh_flags()
        
        # Filter by category if specified
        if category:
            return [flag for flag in self.flags.values() if flag.category == category]
        
        return list(self.flags.values())
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """
        Get a specific feature flag.
        
        Args:
            name: Name of the flag to get
            
        Returns:
            The feature flag, or None if not found
        """
        # Try to refresh if flag not in cache
        if name not in self.flags:
            try:
                key = f"{self.prefix}{name}"
                data = self.redis_client.get(key)
                if data:
                    try:
                        flag_dict = json.loads(data)
                        flag = FeatureFlag.from_dict(flag_dict)
                        self.flags[name] = flag
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding flag data for {key}")
                        return None
                else:
                    return None
            except redis.RedisError:
                return None
        
        return self.flags.get(name)
    
    def register_listener(self, flag_name: str, callback: callable) -> None:
        """
        Register a listener for flag changes.
        
        Args:
            flag_name: Name of the flag to listen for
            callback: Function to call when flag changes
        """
        if flag_name not in self.listeners:
            self.listeners[flag_name] = []
        
        self.listeners[flag_name].append(callback)
        logger.debug(f"Registered listener for flag {flag_name}")
    
    def _notify_listeners(self, flag_name: str, is_enabled: bool, action: str) -> None:
        """
        Notify listeners of a flag change.
        
        Args:
            flag_name: Name of the flag that changed
            is_enabled: New state of the flag
            action: Action that occurred (created, updated, deleted)
        """
        if flag_name in self.listeners:
            for callback in self.listeners[flag_name]:
                try:
                    callback(flag_name, is_enabled, action)
                except Exception as e:
                    logger.error(f"Error in listener for {flag_name}: {str(e)}", exc_info=True)


# Default feature flags for the trading bot
DEFAULT_FLAGS = [
    {
        "name": "enable_trading",
        "enabled": False,
        "description": "Master switch for enabling all trading activity",
        "category": "trading"
    },
    {
        "name": "use_paper_trading",
        "enabled": True,
        "description": "Use paper trading instead of real trading",
        "category": "trading"
    },
    {
        "name": "enable_options_trading",
        "enabled": False,
        "description": "Enable options trading strategies",
        "category": "trading"
    },
    {
        "name": "enable_forex_trading",
        "enabled": False,
        "description": "Enable forex trading strategies",
        "category": "trading"
    },
    {
        "name": "enable_crypto_trading",
        "enabled": False,
        "description": "Enable crypto trading strategies",
        "category": "trading"
    },
    {
        "name": "enable_futures_trading",
        "enabled": False,
        "description": "Enable futures trading strategies",
        "category": "trading"
    },
    {
        "name": "enable_stock_trading",
        "enabled": False,
        "description": "Enable stock trading strategies",
        "category": "trading"
    },
    {
        "name": "use_advanced_risk_management",
        "enabled": True,
        "description": "Use advanced risk management features",
        "category": "risk"
    },
    {
        "name": "enable_stop_loss",
        "enabled": True,
        "description": "Enable automatic stop loss placement",
        "category": "risk"
    },
    {
        "name": "enable_hedging",
        "enabled": False,
        "description": "Enable automatic hedging strategies",
        "category": "risk"
    },
    {
        "name": "enable_telegram_notifications",
        "enabled": True,
        "description": "Send trade notifications via Telegram",
        "category": "notifications"
    },
    {
        "name": "enable_email_notifications",
        "enabled": False,
        "description": "Send trade notifications via email",
        "category": "notifications"
    },
    {
        "name": "debug_mode",
        "enabled": False,
        "description": "Enable debug logging and features",
        "category": "system"
    },
    {
        "name": "maintenance_mode",
        "enabled": False,
        "description": "System maintenance mode - disables trading",
        "category": "system"
    }
]


def init_default_flags(service: FeatureFlagService) -> None:
    """
    Initialize default feature flags if they don't exist.
    
    Args:
        service: FeatureFlagService instance
    """
    for flag_data in DEFAULT_FLAGS:
        name = flag_data["name"]
        if service.get_flag(name) is None:
            service.create_flag(
                name=name,
                enabled=flag_data["enabled"],
                description=flag_data["description"],
                category=flag_data["category"]
            )


# Singleton instance for global access
_instance = None

def get_feature_flag_service() -> FeatureFlagService:
    """
    Get the global FeatureFlagService instance.
    
    Returns:
        FeatureFlagService instance
    """
    global _instance
    if _instance is None:
        # Read config from environment
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        # Create service
        _instance = FeatureFlagService(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_password=redis_password
        )
        
        # Initialize default flags
        init_default_flags(_instance)
    
    return _instance


def is_feature_enabled(name: str, default: bool = False) -> bool:
    """
    Convenience function to check if a feature flag is enabled.
    
    Args:
        name: Name of the flag to check
        default: Default value if flag doesn't exist
        
    Returns:
        True if the flag is enabled, default if not found
    """
    return get_feature_flag_service().is_enabled(name, default) 