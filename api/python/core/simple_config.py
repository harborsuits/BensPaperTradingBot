"""
Simple Configuration Manager

A lightweight configuration loader that validates against JSON Schema
and supports environment variable overrides without external dependencies.
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, Type, Callable, TypeVar
from datetime import datetime, time
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Type for configuration
ConfigDict = Dict[str, Any]
T = TypeVar('T')


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


def load_config(config_path: str) -> ConfigDict:
    """
    Load and validate configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigFileNotFoundError: If config file doesn't exist
        ConfigParseError: If config file cannot be parsed as JSON
        ConfigValidationError: If config doesn't pass basic validation
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
    
    # Validate required fields
    required_fields = [
        "enable_market_regime_system",
        "market_regime_config_path",
        "watched_symbols",
        "data_dir",
        "trading_hours",
        "initial_capital",
        "risk_per_trade",
        "max_open_positions"
    ]
    
    missing_fields = [field for field in required_fields if field not in config_data]
    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        logger.error(error_msg)
        raise ConfigValidationError(error_msg)
    
    # Basic validation of field types and values
    validate_config(config_data)
    
    # Apply environment variable overrides
    apply_env_overrides(config_data)
    
    logger.info(f"Configuration loaded successfully from {config_path}")
    return config_data


def validate_config(config: ConfigDict) -> None:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    errors = []
    
    # Validate boolean fields
    if not isinstance(config.get("enable_market_regime_system"), bool):
        errors.append("enable_market_regime_system must be a boolean")
    
    # Validate string fields
    for field in ["market_regime_config_path", "data_dir"]:
        if not isinstance(config.get(field), str):
            errors.append(f"{field} must be a string")
    
    # Validate watched_symbols
    symbols = config.get("watched_symbols", [])
    if not isinstance(symbols, list) or len(symbols) == 0:
        errors.append("watched_symbols must be a non-empty list")
    elif not all(isinstance(s, str) for s in symbols):
        errors.append("watched_symbols must contain only strings")
    
    # Validate trading_hours
    trading_hours = config.get("trading_hours", {})
    if not isinstance(trading_hours, dict):
        errors.append("trading_hours must be an object")
    else:
        # Check required fields
        for field in ["start", "end", "timezone"]:
            if field not in trading_hours:
                errors.append(f"trading_hours.{field} is required")
        
        # Validate time formats
        time_pattern = r'^([01]\d|2[0-3]):([0-5]\d)$'
        for field in ["start", "end"]:
            if field in trading_hours and not re.match(time_pattern, trading_hours[field]):
                errors.append(f"trading_hours.{field} must be in HH:MM format")
        
        # Validate timezone
        if "timezone" in trading_hours:
            tz = trading_hours["timezone"]
            if not isinstance(tz, str) or not re.match(r'^[A-Za-z_]+/[A-Za-z_]+(/[A-Za-z_]+)?$', tz):
                errors.append("trading_hours.timezone must be a valid IANA timezone (e.g., America/New_York)")
    
    # Validate numeric fields
    if not isinstance(config.get("initial_capital"), (int, float)) or config.get("initial_capital", 0) < 0:
        errors.append("initial_capital must be a non-negative number")
    
    if not isinstance(config.get("risk_per_trade"), (int, float)) or not (0 < config.get("risk_per_trade", 0) < 1):
        errors.append("risk_per_trade must be a number between 0 and 1 (exclusive)")
    
    if not isinstance(config.get("max_open_positions"), int) or config.get("max_open_positions", 0) < 1:
        errors.append("max_open_positions must be a positive integer")
    
    # Report errors
    if errors:
        error_msg = "Configuration validation failed:\n- " + "\n- ".join(errors)
        logger.error(error_msg)
        raise ConfigValidationError(error_msg)


def apply_env_overrides(config: ConfigDict) -> None:
    """
    Apply environment variable overrides to configuration.
    
    Args:
        config: Configuration dictionary to update in place
    """
    overrides_applied = []
    
    # Process top-level fields
    for field_name in config.keys():
        env_name = f"BENBOT_{field_name.upper()}"
        env_value = os.environ.get(env_name)
        
        if env_value is not None:
            try:
                # Convert value to appropriate type based on current value
                current_value = config[field_name]
                
                if isinstance(current_value, bool):
                    typed_value = env_value.lower() in ('true', 'yes', '1', 'y')
                elif isinstance(current_value, int):
                    typed_value = int(env_value)
                elif isinstance(current_value, float):
                    typed_value = float(env_value)
                elif isinstance(current_value, list):
                    typed_value = [v.strip() for v in env_value.split(',')]
                else:
                    # Use string as-is
                    typed_value = env_value
                
                # Update config
                config[field_name] = typed_value
                overrides_applied.append(env_name)
                logger.debug(f"Applied environment override: {env_name}={env_value}")
                
            except Exception as e:
                logger.error(f"Error applying environment override for {env_name}: {str(e)}")
    
    # Process nested fields, specifically for trading_hours
    if "trading_hours" in config and isinstance(config["trading_hours"], dict):
        for sub_field in ["start", "end", "timezone"]:
            env_name = f"BENBOT_TRADING_HOURS_{sub_field.upper()}"
            env_value = os.environ.get(env_name)
            
            if env_value is not None:
                try:
                    config["trading_hours"][sub_field] = env_value
                    overrides_applied.append(env_name)
                    logger.debug(f"Applied environment override: {env_name}={env_value}")
                except Exception as e:
                    logger.error(f"Error applying environment override for {env_name}: {str(e)}")
    
    # Process system_safeguards if present
    if "system_safeguards" in config and isinstance(config["system_safeguards"], dict):
        if "circuit_breakers" in config["system_safeguards"] and isinstance(config["system_safeguards"]["circuit_breakers"], dict):
            for sub_field in ["max_drawdown_percent", "max_daily_loss_percent", "consecutive_loss_count"]:
                env_name = f"BENBOT_SYSTEM_SAFEGUARDS_CIRCUIT_BREAKERS_{sub_field.upper()}"
                env_value = os.environ.get(env_name)
                
                if env_value is not None:
                    try:
                        # Convert to appropriate type
                        if sub_field == "consecutive_loss_count":
                            typed_value = int(env_value)
                        else:
                            typed_value = float(env_value)
                        
                        config["system_safeguards"]["circuit_breakers"][sub_field] = typed_value
                        overrides_applied.append(env_name)
                        logger.debug(f"Applied environment override: {env_name}={env_value}")
                    except Exception as e:
                        logger.error(f"Error applying environment override for {env_name}: {str(e)}")
    
    # Report applied overrides
    if overrides_applied:
        logger.info(f"Applied {len(overrides_applied)} environment variable overrides: {', '.join(overrides_applied)}")


def get_nested_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Path to the value using dot notation (e.g., "trading_hours.start")
        default: Default value if path doesn't exist
        
    Returns:
        Value at the path, or default if not found
    """
    parts = path.split('.')
    current = config
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    
    return current


def parse_time(time_str: str) -> time:
    """
    Parse a time string in HH:MM format to a datetime.time object.
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        datetime.time object
    """
    hours, minutes = map(int, time_str.split(':'))
    return time(hour=hours, minute=minutes)


# Helper functions for type conversion
def to_bool(value: Any) -> bool:
    """Convert a value to boolean"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 'y', 'on')
    return bool(value)


def to_int(value: Any, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """Convert a value to integer with optional bounds checking"""
    result = int(value)
    
    if min_value is not None and result < min_value:
        raise ValueError(f"Value {result} is less than minimum {min_value}")
    
    if max_value is not None and result > max_value:
        raise ValueError(f"Value {result} is greater than maximum {max_value}")
    
    return result


def to_float(value: Any, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    """Convert a value to float with optional bounds checking"""
    result = float(value)
    
    if min_value is not None and result < min_value:
        raise ValueError(f"Value {result} is less than minimum {min_value}")
    
    if max_value is not None and result > max_value:
        raise ValueError(f"Value {result} is greater than maximum {max_value}")
    
    return result


def to_list(value: Any, item_type: Optional[Type[T]] = None, separator: str = ',') -> List[T]:
    """Convert a value to list with optional item type conversion"""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [item.strip() for item in value.split(separator)]
    else:
        items = [value]
    
    if item_type is not None:
        return [item_type(item) for item in items]
    
    return items


# Create a configuration object
_config: Optional[ConfigDict] = None


def get_config() -> ConfigDict:
    """
    Get the loaded configuration.
    
    Returns:
        The loaded configuration dictionary
        
    Raises:
        ConfigError: If configuration is not loaded
    """
    global _config
    
    if _config is None:
        raise ConfigError("Configuration not loaded, call load_config() first")
    
    return _config


def init_config(config_path: str) -> ConfigDict:
    """
    Initialize the global configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration dictionary
    """
    global _config
    
    _config = load_config(config_path)
    return _config
