#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Configuration System - Central configuration management for the trading platform.

This module provides a unified approach to configuration management, replacing the
disparate systems previously used throughout the platform.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import copy
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigurationSection:
    """
    Represents a section of the configuration with nested access capabilities.
    """
    
    def __init__(self, data: Dict[str, Any], name: str, parent: Optional['UnifiedConfig'] = None):
        """
        Initialize a configuration section.
        
        Args:
            data: Configuration data dictionary
            name: Section name
            parent: Parent configuration (for updating)
        """
        self._data = data
        self._name = name
        self._parent = parent
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with nested key support.
        
        Example: config.get('database.connection.host')
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self._data.get(key, default)
        
        parts = key.split('.')
        current = self._data
        
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return default
            current = current[part]
        
        return current.get(parts[-1], default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value with nested key support.
        
        Args:
            key: Configuration key (can be nested with dots)
            value: Value to set
        """
        if '.' not in key:
            self._data[key] = value
            if self._parent:
                self._parent.mark_dirty()
            return
        
        parts = key.split('.')
        current = self._data
        
        # Navigate to the parent of the final key
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Convert to dict if not already
                current[part] = {'value': current[part]}
            current = current[part]
        
        # Set the final value
        current[parts[-1]] = value
        
        # Mark as dirty
        if self._parent:
            self._parent.mark_dirty()
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the configuration section as a dictionary."""
        return copy.deepcopy(self._data)
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return key in self._data
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value by key."""
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return ConfigurationSection(value, f"{self._name}.{key}", self._parent)
            return value
        raise KeyError(f"Configuration key '{key}' not found in section '{self._name}'")


class UnifiedConfig:
    """
    Unified configuration management system for the trading platform.
    
    Features:
    - Hierarchical configuration with dot notation access
    - Configuration overlays (env-specific, user-specific)
    - Auto-reload capability
    - Validation framework
    - Change tracking
    """
    
    def __init__(self, 
                base_path: Optional[str] = None, 
                env: str = "development"):
        """
        Initialize the configuration system.
        
        Args:
            base_path: Base path for configuration files (defaults to config directory)
            env: Environment name (development, staging, production)
        """
        self._config_data: Dict[str, Any] = {}
        self._dirty = False
        self._last_loaded: Optional[datetime] = None
        self._env = env
        self._auto_reload = False
        self._reload_interval = 300  # 5 minutes
        self._validation_rules = {}
        self._change_log = []
        self._load_callbacks = []
        
        # Set base path
        if base_path:
            self._base_path = Path(base_path)
        else:
            # Default to config directory
            this_file = Path(__file__).resolve()
            self._base_path = this_file.parent
        
        # Initialize sections
        self._sections = {}
    
    def load(self, reload: bool = False) -> None:
        """
        Load configuration from files.
        
        Args:
            reload: Whether to force reload even if already loaded
        """
        if self._last_loaded is not None and not reload:
            # Check if auto-reload is enabled and we should reload
            if (self._auto_reload and 
                (datetime.now() - self._last_loaded).total_seconds() > self._reload_interval):
                reload = True
            else:
                logger.debug("Configuration already loaded, skipping")
                return
        
        # Start with a clean slate if reloading
        if reload:
            self._config_data = {}
        
        # Load base configuration
        self._load_base_config()
        
        # Load environment-specific configuration
        self._load_env_config()
        
        # Load local overrides
        self._load_local_overrides()
        
        # Initialize sections
        self._init_sections()
        
        # Mark as cleanly loaded
        self._dirty = False
        self._last_loaded = datetime.now()
        
        # Run validation
        self._validate()
        
        # Call load callbacks
        for callback in self._load_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in configuration load callback: {str(e)}")
        
        logger.info(f"Configuration loaded from {self._base_path} (env: {self._env})")
    
    def _load_base_config(self) -> None:
        """Load base configuration from config.yaml or config.json."""
        yaml_path = self._base_path / "config.yaml"
        json_path = self._base_path / "config.json"
        
        if yaml_path.exists():
            self._load_yaml(yaml_path, self._config_data)
        elif json_path.exists():
            self._load_json(json_path, self._config_data)
        else:
            logger.warning(f"No base configuration found at {yaml_path} or {json_path}")
    
    def _load_env_config(self) -> None:
        """Load environment-specific configuration."""
        yaml_path = self._base_path / f"config.{self._env}.yaml"
        json_path = self._base_path / f"config.{self._env}.json"
        
        if yaml_path.exists():
            self._load_yaml(yaml_path, self._config_data, overlay=True)
        elif json_path.exists():
            self._load_json(json_path, self._config_data, overlay=True)
    
    def _load_local_overrides(self) -> None:
        """Load local configuration overrides."""
        yaml_path = self._base_path / "config.local.yaml"
        json_path = self._base_path / "config.local.json"
        
        if yaml_path.exists():
            self._load_yaml(yaml_path, self._config_data, overlay=True)
        elif json_path.exists():
            self._load_json(json_path, self._config_data, overlay=True)
    
    def _load_yaml(self, path: Path, target: Dict[str, Any], overlay: bool = False) -> None:
        """
        Load YAML configuration file.
        
        Args:
            path: Path to YAML file
            target: Target dictionary to load into
            overlay: Whether to overlay (True) or replace (False) existing config
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not overlay:
                target.clear()
                target.update(data)
            else:
                self._deep_update(target, data)
                
            logger.debug(f"Loaded configuration from {path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {str(e)}")
    
    def _load_json(self, path: Path, target: Dict[str, Any], overlay: bool = False) -> None:
        """
        Load JSON configuration file.
        
        Args:
            path: Path to JSON file
            target: Target dictionary to load into
            overlay: Whether to overlay (True) or replace (False) existing config
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if not overlay:
                target.clear()
                target.update(data)
            else:
                self._deep_update(target, data)
                
            logger.debug(f"Loaded configuration from {path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {str(e)}")
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update a dictionary with another dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    def _init_sections(self) -> None:
        """Initialize configuration sections."""
        self._sections = {}
        
        # Create sections for top-level keys
        for key, value in self._config_data.items():
            if isinstance(value, dict):
                self._sections[key] = ConfigurationSection(value, key, self)
    
    def get_section(self, name: str) -> Optional[ConfigurationSection]:
        """
        Get a configuration section.
        
        Args:
            name: Section name
            
        Returns:
            Configuration section or None if not found
        """
        if name in self._sections:
            return self._sections[name]
        
        if '.' in name:
            # Handle nested sections
            parts = name.split('.')
            current = self._config_data
            
            for part in parts:
                if part not in current or not isinstance(current[part], dict):
                    return None
                current = current[part]
            
            return ConfigurationSection(current, name, self)
        
        return None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with nested key support.
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self._config_data.get(key, default)
        
        section_name, sub_key = key.split('.', 1)
        section = self.get_section(section_name)
        
        if section:
            return section.get(sub_key, default)
        
        return default
    
    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set a configuration value with nested key support.
        
        Args:
            key: Configuration key (can be nested with dots)
            value: Value to set
            persist: Whether to persist the change to disk
        """
        # Store the old value for change log
        old_value = self.get(key)
        
        if '.' not in key:
            self._config_data[key] = value
            
            # Update section if it's a dict
            if isinstance(value, dict):
                self._sections[key] = ConfigurationSection(value, key, self)
                
            self._dirty = True
        else:
            section_name, sub_key = key.split('.', 1)
            
            if section_name not in self._config_data:
                self._config_data[section_name] = {}
                self._sections[section_name] = ConfigurationSection(
                    self._config_data[section_name], section_name, self
                )
            
            section = self.get_section(section_name)
            if section:
                section.set(sub_key, value)
                self._dirty = True
        
        # Log the change
        self._change_log.append({
            'key': key,
            'old_value': old_value,
            'new_value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Persist if requested
        if persist:
            self.save()
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to disk.
        
        Args:
            path: Optional path to save to (defaults to config.local.yaml)
        """
        if path is None:
            path = self._base_path / "config.local.yaml"
        
        # Validate before saving
        validation_errors = self._validate()
        if validation_errors:
            error_msg = "\n".join([f"{k}: {v}" for k, v in validation_errors.items()])
            logger.error(f"Cannot save configuration due to validation errors:\n{error_msg}")
            return
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump(self._config_data, f, indent=2)
            else:
                with open(path, 'w') as f:
                    yaml.dump(self._config_data, f, default_flow_style=False)
            
            self._dirty = False
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {str(e)}")
    
    def mark_dirty(self) -> None:
        """Mark the configuration as dirty (modified)."""
        self._dirty = True
    
    def is_dirty(self) -> bool:
        """Check if the configuration has unsaved changes."""
        return self._dirty
    
    def add_validation_rule(self, key: str, rule: Callable[[Any], bool], 
                          error_message: str) -> None:
        """
        Add a validation rule for a configuration key.
        
        Args:
            key: Configuration key to validate
            rule: Validation function (returns True if valid)
            error_message: Error message to display if validation fails
        """
        self._validation_rules[key] = {
            'rule': rule,
            'error_message': error_message
        }
    
    def _validate(self) -> Dict[str, str]:
        """
        Validate the configuration against defined rules.
        
        Returns:
            Dictionary mapping keys to error messages for failed validations
        """
        errors = {}
        
        for key, rule_info in self._validation_rules.items():
            value = self.get(key)
            rule = rule_info['rule']
            
            if not rule(value):
                errors[key] = rule_info['error_message']
                logger.warning(f"Configuration validation failed for {key}: {rule_info['error_message']}")
        
        return errors
    
    def add_load_callback(self, callback: Callable[['UnifiedConfig'], None]) -> None:
        """
        Add a callback to be called when configuration is loaded.
        
        Args:
            callback: Function to call after configuration is loaded
        """
        self._load_callbacks.append(callback)
    
    def set_auto_reload(self, enabled: bool, interval: int = 300) -> None:
        """
        Enable or disable automatic configuration reloading.
        
        Args:
            enabled: Whether to enable auto-reload
            interval: Reload interval in seconds (default: 5 minutes)
        """
        self._auto_reload = enabled
        self._reload_interval = interval
        
        logger.info(f"Configuration auto-reload {'enabled' if enabled else 'disabled'}" +
                   (f" (interval: {interval}s)" if enabled else ""))
    
    def get_change_log(self) -> List[Dict[str, Any]]:
        """Get the configuration change log."""
        return copy.deepcopy(self._change_log)
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return copy.deepcopy(self._config_data)
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value or section by key."""
        if key in self._sections:
            return self._sections[key]
            
        if key in self._config_data:
            value = self._config_data[key]
            if isinstance(value, dict):
                # Create section on-demand
                self._sections[key] = ConfigurationSection(value, key, self)
                return self._sections[key]
            return value
            
        raise KeyError(f"Configuration key '{key}' not found")
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return key in self._config_data


# Create a global configuration instance
config = UnifiedConfig()

# Helper functions
def load_config(reload: bool = False) -> None:
    """
    Load configuration.
    
    Args:
        reload: Whether to force reload
    """
    config.load(reload)

def get_config() -> UnifiedConfig:
    """
    Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    if config._last_loaded is None:
        config.load()
    return config

def get(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    return get_config().get(key, default)

def set(key: str, value: Any, persist: bool = False) -> None:
    """
    Set a configuration value.
    
    Args:
        key: Configuration key
        value: Value to set
        persist: Whether to persist the change to disk
    """
    get_config().set(key, value, persist)
