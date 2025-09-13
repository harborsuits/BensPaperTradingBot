"""
Configuration Manager

Centralized configuration management system for the trading bot and dashboard.
Handles loading, validating, and persisting configuration settings.
"""

import os
import json
import yaml
import logging
import copy
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

class ConfigManager:
    """
    Manages configuration settings for the trading bot and dashboard.
    
    This class provides a unified interface for loading, validating, and 
    persisting configuration settings, with support for:
    - Multiple configuration formats (JSON, YAML)
    - Configuration schema validation
    - Environment variable overrides
    - User-specific settings
    - Configuration change history
    """
    
    def __init__(self, 
                config_dir: str = None, 
                default_config_path: str = None,
                schema_path: str = None,
                environment_prefix: str = "TRADING_BOT_"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            default_config_path: Path to default configuration file
            schema_path: Path to JSON schema for validation
            environment_prefix: Prefix for environment variables to override settings
        """
        self.logger = logging.getLogger("ConfigManager")
        self.logger.setLevel(logging.INFO)
        
        # Set configuration paths
        self.config_dir = config_dir or os.path.expanduser("~/.trading_bot")
        self.default_config_path = default_config_path
        self.schema_path = schema_path
        self.environment_prefix = environment_prefix
        
        # Create configuration directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize configuration properties
        self.config = {}
        self.base_config = {}
        self.user_config = {}
        self.config_loaded = False
        self.config_history = []
        self.change_callbacks = []
        
        # Load default configuration if provided
        if default_config_path and os.path.exists(default_config_path):
            self._load_default_config()
            
        # Load JSON schema if provided
        self.schema = None
        if schema_path and os.path.exists(schema_path):
            self._load_schema()
        
        # Initialize configuration watchers
        self.watch_thread = None
        self.watch_interval = 60  # seconds
        self.watching = False
    
    def _load_default_config(self) -> None:
        """Load default configuration from the specified file."""
        try:
            file_ext = os.path.splitext(self.default_config_path)[1].lower()
            
            with open(self.default_config_path, 'r') as f:
                if file_ext == '.json':
                    self.base_config = json.load(f)
                elif file_ext in ('.yaml', '.yml'):
                    self.base_config = yaml.safe_load(f)
                else:
                    self.logger.warning(f"Unsupported config format: {file_ext}")
                    return
                
            self.logger.info(f"Loaded default configuration from {self.default_config_path}")
        except Exception as e:
            self.logger.error(f"Error loading default configuration: {e}")
    
    def _load_schema(self) -> None:
        """Load JSON schema for configuration validation."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
            self.logger.info(f"Loaded configuration schema from {self.schema_path}")
        except Exception as e:
            self.logger.error(f"Error loading schema: {e}")
    
    def load_config(self, config_name: str = "config") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Dictionary with loaded configuration
        """
        # Try to load user configuration
        self.user_config = {}
        config_paths = [
            os.path.join(self.config_dir, f"{config_name}.json"),
            os.path.join(self.config_dir, f"{config_name}.yaml"),
            os.path.join(self.config_dir, f"{config_name}.yml")
        ]
        
        loaded = False
        for path in config_paths:
            if os.path.exists(path):
                try:
                    file_ext = os.path.splitext(path)[1].lower()
                    with open(path, 'r') as f:
                        if file_ext == '.json':
                            self.user_config = json.load(f)
                        else:  # yaml or yml
                            self.user_config = yaml.safe_load(f)
                    
                    self.logger.info(f"Loaded user configuration from {path}")
                    loaded = True
                    break
                except Exception as e:
                    self.logger.error(f"Error loading user configuration from {path}: {e}")
        
        if not loaded:
            self.logger.warning(f"No user configuration found, using defaults")
        
        # Merge configurations
        self.config = copy.deepcopy(self.base_config)
        self._merge_config(self.config, self.user_config)
        
        # Apply environment variable overrides
        self._apply_environment_overrides()
        
        # Validate configuration
        if self.schema:
            valid, errors = self._validate_config()
            if not valid:
                self.logger.warning(f"Configuration validation failed: {errors}")
            
        self.config_loaded = True
        return self.config
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary (modified in place)
            source: Source dictionary
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_vars = {key: value for key, value in os.environ.items() 
                   if key.startswith(self.environment_prefix)}
        
        override_count = 0
        for key, value in env_vars.items():
            config_key = key[len(self.environment_prefix):].lower()
            config_path = config_key.split('__')
            
            # Navigate to the target configuration
            target = self.config
            for part in config_path[:-1]:
                if part not in target:
                    target[part] = {}
                elif not isinstance(target[part], dict):
                    # Cannot override a non-dict with a dict
                    self.logger.warning(f"Cannot apply environment override for {key}: path contains non-dict value")
                    break
                target = target[part]
            else:
                # Set the value (convert to appropriate type)
                last_key = config_path[-1]
                parsed_value = self._parse_env_value(value)
                target[last_key] = parsed_value
                override_count += 1
        
        if override_count > 0:
            self.logger.info(f"Applied {override_count} environment variable overrides")
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value as appropriate type
        """
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try to parse as boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate_config(self) -> tuple:
        """
        Validate configuration against JSON schema.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self.schema:
            return True, None
        
        try:
            import jsonschema
            jsonschema.validate(instance=self.config, schema=self.schema)
            return True, None
        except ImportError:
            self.logger.warning("jsonschema package not available, skipping validation")
            return True, "jsonschema package not available"
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)
    
    def save_config(self, config_name: str = "config", format: str = "json") -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_name: Name of configuration file (without extension)
            format: File format (json or yaml)
            
        Returns:
            True if successfully saved, False otherwise
        """
        if format.lower() not in ('json', 'yaml', 'yml'):
            self.logger.error(f"Unsupported format: {format}")
            return False
        
        file_extension = '.json' if format.lower() == 'json' else '.yaml'
        config_path = os.path.join(self.config_dir, f"{config_name}{file_extension}")
        
        # Add to history before saving
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'config': copy.deepcopy(self.config)
        })
        if len(self.config_history) > 10:
            self.config_history = self.config_history[-10:]
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(self.config, f, indent=2)
                else:
                    yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info(f"Saved configuration to {config_path}")
            
            # Trigger change callbacks
            for callback in self.change_callbacks:
                try:
                    callback(self.config)
                except Exception as e:
                    self.logger.error(f"Error in configuration change callback: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary with current configuration
        """
        if not self.config_loaded:
            self.load_config()
        
        return copy.deepcopy(self.config)
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        if not self.config_loaded:
            self.load_config()
        
        # Navigate through the config hierarchy
        parts = key_path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set_value(self, key_path: str, value: Any, save: bool = True) -> bool:
        """
        Set configuration value by key path.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: New value to set
            save: Whether to save configuration after setting value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config_loaded:
            self.load_config()
        
        # Navigate through the config hierarchy
        parts = key_path.split('.')
        target = self.config
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            elif not isinstance(target[part], dict):
                self.logger.error(f"Cannot set {key_path}: path contains non-dict value")
                return False
            
            target = target[part]
        
        # Set the value
        target[parts[-1]] = value
        
        # Save if requested
        if save:
            return self.save_config()
        
        return True
    
    def register_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback to be called when configuration changes.
        
        Args:
            callback: Function to call with new configuration
        """
        self.change_callbacks.append(callback)
    
    def start_watching(self, interval: int = 60) -> bool:
        """
        Start watching configuration file for changes.
        
        Args:
            interval: Interval in seconds between checks
            
        Returns:
            True if watching started, False otherwise
        """
        if self.watching:
            return True
        
        try:
            import threading
            
            self.watch_interval = interval
            self.watching = True
            
            def watch_thread():
                last_modified = {}
                
                while self.watching:
                    # Check each possible config file
                    config_name = "config"  # TODO: Make this configurable
                    config_paths = [
                        os.path.join(self.config_dir, f"{config_name}.json"),
                        os.path.join(self.config_dir, f"{config_name}.yaml"),
                        os.path.join(self.config_dir, f"{config_name}.yml")
                    ]
                    
                    for path in config_paths:
                        if os.path.exists(path):
                            modified = os.path.getmtime(path)
                            
                            if path in last_modified and modified > last_modified[path]:
                                self.logger.info(f"Configuration file changed: {path}")
                                self.load_config()
                                
                                # Trigger change callbacks
                                for callback in self.change_callbacks:
                                    try:
                                        callback(self.config)
                                    except Exception as e:
                                        self.logger.error(f"Error in configuration change callback: {e}")
                            
                            last_modified[path] = modified
                    
                    # Wait for next check
                    time.sleep(self.watch_interval)
            
            self.watch_thread = threading.Thread(target=watch_thread, daemon=True)
            self.watch_thread.start()
            
            self.logger.info(f"Started watching configuration files (interval: {interval}s)")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting configuration watcher: {e}")
            self.watching = False
            return False
    
    def stop_watching(self) -> None:
        """Stop watching configuration file for changes."""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1.0)
            self.watch_thread = None
            self.logger.info("Stopped watching configuration files")
    
    def get_config_history(self) -> List[Dict[str, Any]]:
        """
        Get configuration change history.
        
        Returns:
            List of historical configurations with timestamps
        """
        return self.config_history
    
    def reset_to_defaults(self, save: bool = True) -> bool:
        """
        Reset configuration to default values.
        
        Args:
            save: Whether to save configuration after reset
            
        Returns:
            True if successful, False otherwise
        """
        if not self.base_config:
            self.logger.warning("No default configuration available")
            return False
        
        self.config = copy.deepcopy(self.base_config)
        self.logger.info("Reset configuration to defaults")
        
        if save:
            return self.save_config()
        
        return True

# Default configuration for trading dashboard
DEFAULT_DASHBOARD_CONFIG = {
    "api": {
        "url": "http://localhost:5000",
        "timeout": 5,
        "retry_count": 3,
        "authentication": {
            "enabled": False,
            "token": ""
        }
    },
    "dashboard": {
        "refresh_interval": 10,
        "color_theme": "dark",
        "layout": "default",
        "panels": {
            "summary": True,
            "open_positions": True,
            "recent_trades": True,
            "recommendations": True,
            "statistics": True,
            "charts": True,
            "notifications": True,
            "commands": True
        }
    },
    "notifications": {
        "enabled": True,
        "min_level": "INFO",
        "desktop": {
            "enabled": True
        },
        "email": {
            "enabled": False,
            "recipients": []
        },
        "slack": {
            "enabled": False,
            "channel": "#trading-alerts"
        }
    },
    "charts": {
        "equity_curve": {
            "enabled": True,
            "days": 30,
            "include_deposit_withdrawals": True
        },
        "win_loss": {
            "enabled": True,
            "count": 50
        },
        "strategy_performance": {
            "enabled": True,
            "top_count": 5
        }
    },
    "logging": {
        "level": "INFO",
        "file": {
            "enabled": True,
            "path": "dashboard.log",
            "max_size_mb": 10,
            "backup_count": 3
        },
        "console": {
            "enabled": True,
            "level": "INFO"
        }
    }
}

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration manager
    config_dir = os.path.expanduser("~/.trading_bot")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create default config file if it doesn't exist
    default_config_path = os.path.join(config_dir, "default_config.json")
    if not os.path.exists(default_config_path):
        with open(default_config_path, 'w') as f:
            json.dump(DEFAULT_DASHBOARD_CONFIG, f, indent=2)
    
    config_manager = ConfigManager(
        config_dir=config_dir,
        default_config_path=default_config_path
    )
    
    # Load configuration
    config = config_manager.load_config()
    
    # Example: get specific value
    api_url = config_manager.get_value("api.url")
    print(f"API URL: {api_url}")
    
    # Example: set specific value
    config_manager.set_value("dashboard.refresh_interval", 15)
    
    # Example: save configuration
    config_manager.save_config() 