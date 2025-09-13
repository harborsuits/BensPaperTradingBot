"""
Configuration Manager

This module provides functionality for loading, validating, and accessing BensBot configuration.
It handles JSON schema validation, environment variable overrides, and provides typed access
to configuration values through Pydantic models.
"""

import os
import json
import logging
import jsonschema
from typing import Dict, Any, Optional, Type, TypeVar, cast, List, Union
from pathlib import Path

from trading_bot.core.models.config_model import ConfigModel, create_default_config

logger = logging.getLogger(__name__)

# Type variable for config models
T = TypeVar('T', bound=ConfigModel)


class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """Raised when configuration file is not found"""
    pass


class ConfigParseError(ConfigError):
    """Raised when configuration file cannot be parsed"""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails"""
    pass


class ConfigManager:
    """
    Manager for BensBot configuration.
    
    Handles loading, validation, and environment variable overrides
    for the trading system configuration.
    """
    
    def __init__(self):
        """Initialize the configuration manager"""
        self._config: Optional[ConfigModel] = None
        self._raw_config: Optional[Dict[str, Any]] = None
        self._schema_path = Path(__file__).parent.parent.parent / "config" / "system_config.schema.json"
    
    def load_config(self, config_path: str) -> ConfigModel:
        """
        Load and validate configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated ConfigModel instance
            
        Raises:
            ConfigFileNotFoundError: If config file doesn't exist
            ConfigParseError: If config file cannot be parsed as JSON
            ConfigValidationError: If config doesn't conform to schema
        """
        path = Path(config_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise ConfigFileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Read and parse JSON
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
                
            logger.debug(f"Loaded raw configuration from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            raise ConfigParseError(f"Error parsing configuration file: {str(e)}")
        
        # Validate against schema if schema file exists
        if self._schema_path.exists():
            try:
                with open(self._schema_path, 'r') as f:
                    schema = json.load(f)
                
                jsonschema.validate(instance=config_data, schema=schema)
                logger.debug("Configuration validated against schema")
            except jsonschema.exceptions.ValidationError as e:
                logger.error(f"Configuration validation error: {str(e)}")
                raise ConfigValidationError(f"Configuration validation error: {str(e)}")
        else:
            logger.warning(f"Schema file not found: {self._schema_path}, skipping schema validation")
        
        # Store raw config
        self._raw_config = config_data
        
        # Parse with Pydantic
        try:
            self._config = ConfigModel(**config_data)
            logger.debug("Configuration parsed into Pydantic model")
        except Exception as e:
            logger.error(f"Error validating configuration with Pydantic: {str(e)}")
            raise ConfigValidationError(f"Error validating configuration: {str(e)}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        return self._config
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should be prefixed with BENBOT_
        and use uppercase with underscores, e.g., BENBOT_INITIAL_CAPITAL.
        
        Nested fields use dot notation, e.g., BENBOT_TRADING_HOURS_START.
        """
        if not self._config:
            logger.warning("Cannot apply environment overrides: configuration not loaded")
            return
        
        # Get all fields from the model
        overrides_applied = []
        
        # Process simple top-level fields
        for field_name in self._config.__fields__:
            env_name = f"BENBOT_{field_name.upper()}"
            env_value = os.environ.get(env_name)
            
            if env_value is not None:
                logger.debug(f"Found environment override: {env_name}={env_value}")
                self._override_field(field_name, env_value)
                overrides_applied.append(env_name)
        
        # Process nested fields, specifically for trading_hours
        for nested_field in ['trading_hours']:
            if hasattr(self._config, nested_field):
                nested_obj = getattr(self._config, nested_field)
                for sub_field in nested_obj.__fields__:
                    env_name = f"BENBOT_{nested_field.upper()}_{sub_field.upper()}"
                    env_value = os.environ.get(env_name)
                    
                    if env_value is not None:
                        logger.debug(f"Found environment override for nested field: {env_name}={env_value}")
                        # Set attribute on nested object
                        try:
                            # Convert the value to appropriate type
                            field_type = type(getattr(nested_obj, sub_field))
                            if field_type == bool:
                                typed_value = env_value.lower() in ('true', 'yes', '1', 'y')
                            elif field_type == int:
                                typed_value = int(env_value)
                            elif field_type == float:
                                typed_value = float(env_value)
                            else:
                                typed_value = env_value
                            
                            setattr(nested_obj, sub_field, typed_value)
                            overrides_applied.append(env_name)
                        except Exception as e:
                            logger.error(f"Error applying environment override for {env_name}: {str(e)}")
        
        # Process notifications hierarchy
        self._process_notification_overrides(overrides_applied)
        
        # Report applied overrides
        if overrides_applied:
            logger.info(f"Applied {len(overrides_applied)} environment variable overrides: {', '.join(overrides_applied)}")
    
    def _process_notification_overrides(self, overrides_applied: List[str]) -> None:
        """Process notification-specific environment overrides"""
        if not self._config or not hasattr(self._config, 'notification_settings'):
            return
        
        # Handle email alerts
        email_prefix = "BENBOT_NOTIFICATION_SETTINGS_EMAIL_ALERTS"
        email_enabled = os.environ.get(f"{email_prefix}_ENABLED")
        if email_enabled is not None:
            enabled = email_enabled.lower() in ('true', 'yes', '1', 'y')
            self._config.notification_settings.email_alerts.enabled = enabled
            overrides_applied.append(f"{email_prefix}_ENABLED")
        
        email = os.environ.get(f"{email_prefix}_EMAIL")
        if email is not None:
            self._config.notification_settings.email_alerts.email = email
            overrides_applied.append(f"{email_prefix}_EMAIL")
        
        # Handle SMS alerts
        sms_prefix = "BENBOT_NOTIFICATION_SETTINGS_SMS_ALERTS"
        sms_enabled = os.environ.get(f"{sms_prefix}_ENABLED")
        if sms_enabled is not None:
            enabled = sms_enabled.lower() in ('true', 'yes', '1', 'y')
            self._config.notification_settings.sms_alerts.enabled = enabled
            overrides_applied.append(f"{sms_prefix}_ENABLED")
        
        phone = os.environ.get(f"{sms_prefix}_PHONE_NUMBER")
        if phone is not None:
            self._config.notification_settings.sms_alerts.phone_number = phone
            overrides_applied.append(f"{sms_prefix}_PHONE_NUMBER")
    
    def _override_field(self, field_name: str, env_value: str) -> None:
        """
        Override a field in the config model with a value from an environment variable.
        
        Args:
            field_name: Name of the field to override
            env_value: String value from environment variable
        """
        if not self._config:
            return
        
        try:
            # Get current value and infer type
            current_value = getattr(self._config, field_name)
            field_type = type(current_value)
            
            # Convert the string value to the appropriate type
            if field_type == bool:
                typed_value = env_value.lower() in ('true', 'yes', '1', 'y')
            elif field_type == int:
                typed_value = int(env_value)
            elif field_type == float:
                typed_value = float(env_value)
            elif field_type == list:
                # Split comma-separated values
                typed_value = [v.strip() for v in env_value.split(',')]
            else:
                # Use string as-is for string or string-based enums
                typed_value = env_value
            
            # Set the value
            setattr(self._config, field_name, typed_value)
            
        except Exception as e:
            logger.error(f"Error applying environment override for {field_name}: {str(e)}")
    
    def get_config(self) -> ConfigModel:
        """
        Get the loaded configuration model.
        
        Returns:
            The loaded ConfigModel instance
            
        Raises:
            ConfigError: If configuration is not loaded
        """
        if not self._config:
            raise ConfigError("Configuration not loaded, call load_config() first")
        
        return self._config
    
    def get_raw_config(self) -> Dict[str, Any]:
        """
        Get the raw configuration dictionary.
        
        Returns:
            The raw configuration dictionary
            
        Raises:
            ConfigError: If configuration is not loaded
        """
        if not self._raw_config:
            raise ConfigError("Raw configuration not available, call load_config() first")
        
        return self._raw_config
    
    def dump_config(self) -> str:
        """
        Dump the current configuration as a JSON string.
        
        Returns:
            JSON string of the current configuration
            
        Raises:
            ConfigError: If configuration is not loaded
        """
        if not self._config:
            raise ConfigError("Configuration not loaded, call load_config() first")
        
        return self._config.json(indent=2)


# Singleton instance
_config_manager = ConfigManager()


def load_config(config_path: str) -> ConfigModel:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated ConfigModel instance
    """
    return _config_manager.load_config(config_path)


def get_config() -> ConfigModel:
    """
    Get the loaded configuration.
    
    Returns:
        The loaded ConfigModel instance
        
    Raises:
        ConfigError: If configuration is not loaded
    """
    return _config_manager.get_config()


def dump_config() -> str:
    """
    Dump the current configuration as a JSON string.
    
    Returns:
        JSON string of the current configuration
    """
    return _config_manager.dump_config()


def create_sample_config(output_path: str) -> None:
    """
    Create a sample configuration file with comments.
    
    Args:
        output_path: Path to write the sample configuration file
    """
    # Create a default config
    config = create_default_config()
    
    # Convert to dict for easier manipulation
    config_dict = config.dict()
    
    # Add comments for each section
    commented_config = {
        "__comment": "BensBot Trading System Configuration",
        "__comment_enable_market_regime_system": "Enable market regime detection system",
        "enable_market_regime_system": config_dict["enable_market_regime_system"],
        
        "__comment_market_regime_config_path": "Path to market regime configuration file",
        "market_regime_config_path": config_dict["market_regime_config_path"],
        
        "__comment_watched_symbols": "List of symbols to monitor and trade",
        "watched_symbols": config_dict["watched_symbols"],
        
        "__comment_data_dir": "Directory to store data files",
        "data_dir": config_dict["data_dir"],
        
        "__comment_trading_hours": "Trading hours configuration",
        "trading_hours": config_dict["trading_hours"],
        
        "__comment_initial_capital": "Initial capital for trading ($)",
        "initial_capital": config_dict["initial_capital"],
        
        "__comment_risk_per_trade": "Maximum risk per trade as a decimal (e.g., 0.02 for 2%)",
        "risk_per_trade": config_dict["risk_per_trade"],
        
        "__comment_max_open_positions": "Maximum number of open positions at any time",
        "max_open_positions": config_dict["max_open_positions"],
        
        "__comment_broker_config_path": "Path to broker configuration file",
        "broker_config_path": config_dict["broker_config_path"],
        
        "__comment_market_data_config_path": "Path to market data configuration file",
        "market_data_config_path": config_dict["market_data_config_path"],
        
        "__comment_log_level": "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        "log_level": config_dict["log_level"],
        
        "__comment_notification_settings": "Notification settings for alerts",
        "notification_settings": config_dict["notification_settings"],
        
        "__comment_performance_tracking": "Performance tracking settings",
        "performance_tracking": config_dict["performance_tracking"],
        
        "__comment_system_safeguards": "System safeguard settings",
        "system_safeguards": config_dict["system_safeguards"]
    }
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(commented_config, f, indent=2, default=str)
    
    print(f"Created sample configuration file at: {output_path}")
