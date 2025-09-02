"""Configuration system for EvoTrader with validation and overrides."""

import yaml
import os
import logging
import json
from typing import Dict, Any, Optional, List
from copy import deepcopy


class ConfigManager:
    """
    Manages configuration loading, validation and overrides.
    
    Features:
    - Load configs from YAML files or dictionaries
    - Deep merge of multiple configs
    - Validation against schema
    - Environment variable overrides
    - Default config values
    - Config serialization
    """
    
    def __init__(self, default_config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            default_config_path: Path to the default config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config: Dict[str, Any] = {}
        
        if default_config_path:
            self.load_config(default_config_path)
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            Dict: The loaded configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        self.logger.info(f"Loading configuration from {path}")
        with open(path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            
        if not loaded_config:
            self.logger.warning(f"Empty or invalid configuration in {path}")
            return {}
            
        # Merge with existing config
        if self.config:
            self.config = self._deep_merge(self.config, loaded_config)
        else:
            self.config = loaded_config
            
        return self.config
    
    def update_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with values from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Dict: The updated configuration
        """
        if not config_dict:
            return self.config
            
        self.config = self._deep_merge(self.config, config_dict)
        return self.config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-notation path.
        
        Args:
            key_path: Dot-notation path to the config value (e.g., 'simulation.seed')
            default: Default value if the path doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using a dot-notation path.
        
        Args:
            key_path: Dot-notation path to the config value (e.g., 'simulation.seed')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the nested dict
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        self.logger.info(f"Configuration saved to {path}")
    
    def load_env_overrides(self, prefix: str = "EVOTRADER_") -> None:
        """
        Override configuration values from environment variables.
        
        Variables should be named like PREFIX_KEY__SUBKEY__SUBSUBKEY
        
        Args:
            prefix: Environment variable prefix to look for
        """
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Remove prefix and split by double underscore
                key_path = env_var[len(prefix):].lower().replace('__', '.')
                
                # Try to parse as JSON first
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, use the string value
                    parsed_value = value
                
                self.set(key_path, parsed_value)
                self.logger.debug(f"Overrode {key_path} from environment")
    
    def validate(self, required_keys: List[str]) -> bool:
        """
        Validate that all required keys exist in the configuration.
        
        Args:
            required_keys: List of dot-notation paths that must exist
            
        Returns:
            bool: True if valid, False otherwise
        """
        missing = []
        for key_path in required_keys:
            if self.get(key_path) is None:
                missing.append(key_path)
        
        if missing:
            self.logger.error(f"Missing required configuration: {', '.join(missing)}")
            return False
            
        return True
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with values from override taking precedence.
        
        Args:
            base: Base dictionary
            override: Dictionary with values to override
            
        Returns:
            Dict: Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively merge nested dictionaries
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                # Override or add the value
                result[key] = deepcopy(value)
                
        return result


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return as a dict.
    Legacy function for backward compatibility.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_config(path)
