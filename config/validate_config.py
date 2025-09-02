#!/usr/bin/env python3
"""
Configuration Validator

A standalone script to validate BensBot configuration files
and check environment variable overrides.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions
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

def load_json(file_path):
    """Load JSON from file path"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigFileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ConfigParseError(f"Invalid JSON in {file_path}: {str(e)}")

def validate_config(config: Dict[str, Any]) -> None:
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
        raise ConfigValidationError(error_msg)

def validate_against_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Simple JSON Schema validation.
    
    Args:
        config: Configuration dictionary
        schema: JSON Schema dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required properties
    required = schema.get("required", [])
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check property types and constraints
    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        if prop_name in config:
            value = config[prop_name]
            
            # Type validation
            prop_type = prop_schema.get("type")
            if prop_type == "string" and not isinstance(value, str):
                errors.append(f"{prop_name} must be a string")
            elif prop_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{prop_name} must be a number")
            elif prop_type == "integer" and not isinstance(value, int):
                errors.append(f"{prop_name} must be an integer")
            elif prop_type == "boolean" and not isinstance(value, bool):
                errors.append(f"{prop_name} must be a boolean")
            elif prop_type == "array" and not isinstance(value, list):
                errors.append(f"{prop_name} must be an array")
            elif prop_type == "object" and not isinstance(value, dict):
                errors.append(f"{prop_name} must be an object")
            
            # String constraints
            if prop_type == "string":
                if "pattern" in prop_schema and not re.match(prop_schema["pattern"], value):
                    errors.append(f"{prop_name} does not match pattern {prop_schema['pattern']}")
                
                if "minLength" in prop_schema and len(value) < prop_schema["minLength"]:
                    errors.append(f"{prop_name} is too short (minimum length: {prop_schema['minLength']})")
                
                if "maxLength" in prop_schema and len(value) > prop_schema["maxLength"]:
                    errors.append(f"{prop_name} is too long (maximum length: {prop_schema['maxLength']})")
            
            # Numeric constraints
            elif prop_type in ["number", "integer"]:
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    errors.append(f"{prop_name} is too small (minimum: {prop_schema['minimum']})")
                
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    errors.append(f"{prop_name} is too large (maximum: {prop_schema['maximum']})")
            
            # Array constraints
            elif prop_type == "array":
                if "minItems" in prop_schema and len(value) < prop_schema["minItems"]:
                    errors.append(f"{prop_name} has too few items (minimum: {prop_schema['minItems']})")
                
                if "maxItems" in prop_schema and len(value) > prop_schema["maxItems"]:
                    errors.append(f"{prop_name} has too many items (maximum: {prop_schema['maxItems']})")
                
                # Validate items if specified
                if "items" in prop_schema and len(value) > 0:
                    item_schema = prop_schema["items"]
                    for i, item in enumerate(value):
                        if "type" in item_schema:
                            if item_schema["type"] == "string" and not isinstance(item, str):
                                errors.append(f"{prop_name}[{i}] must be a string")
                            elif item_schema["type"] == "number" and not isinstance(item, (int, float)):
                                errors.append(f"{prop_name}[{i}] must be a number")
                            elif item_schema["type"] == "integer" and not isinstance(item, int):
                                errors.append(f"{prop_name}[{i}] must be an integer")
            
            # Object constraints - recursively validate properties
            elif prop_type == "object" and "properties" in prop_schema:
                nested_errors = validate_against_schema(value, prop_schema)
                for error in nested_errors:
                    errors.append(f"{prop_name}.{error}")
    
    return errors

def find_env_overrides(prefix="BENBOT_"):
    """Find environment variables with the given prefix"""
    overrides = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            overrides[config_key] = value
    return overrides

def print_env_usage():
    """Print information about using environment variables"""
    print("\nEnvironment Variable Usage:")
    print("--------------------------")
    print("BensBot supports configuration overrides via environment variables.")
    print("Example:")
    print("  export BENBOT_INITIAL_CAPITAL=20000")
    print("  export BENBOT_RISK_PER_TRADE=0.05")
    print("  export BENBOT_TRADING_HOURS_START=10:00")
    print("  export BENBOT_WATCHED_SYMBOLS=SPY,QQQ,AAPL")
    print("\nNested properties use underscores:")
    print("  BENBOT_TRADING_HOURS_START -> trading_hours.start")
    print("  BENBOT_SYSTEM_SAFEGUARDS_CIRCUIT_BREAKERS_MAX_DRAWDOWN_PERCENT -> system_safeguards.circuit_breakers.max_drawdown_percent")
    print("\nLists are comma-separated:")
    print("  BENBOT_WATCHED_SYMBOLS=SPY,QQQ,AAPL")

def check_referenced_files(config):
    """Check that referenced files exist"""
    files_to_check = []
    
    # Add referenced config files
    if 'market_regime_config_path' in config:
        files_to_check.append(config['market_regime_config_path'])
    if 'broker_config_path' in config:
        files_to_check.append(config['broker_config_path'])
    if 'market_data_config_path' in config:
        files_to_check.append(config['market_data_config_path'])
    
    # Check if data directory exists
    if 'data_dir' in config:
        if not os.path.exists(config['data_dir']):
            print(f"⚠️ Warning: Data directory does not exist: {config['data_dir']}")
            print(f"   It will be created when the system runs.")
    
    # Check referenced files
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️ Warning: The following referenced files are missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All referenced configuration files exist")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BensBot Configuration Validator")
    parser.add_argument("--config", default="config/system_config.json", help="Path to config file")
    parser.add_argument("--schema", default="config/system_config.schema.json", help="Path to schema file")
    parser.add_argument("--show-env", action="store_true", help="Show environment variable usage")
    parser.add_argument("--env-prefix", default="BENBOT_", help="Environment variable prefix")
    args = parser.parse_args()
    
    print("\n=== BensBot Configuration Validator ===\n")
    
    try:
        # Load and validate schema
        try:
            schema = load_json(args.schema)
            if "$schema" not in schema:
                print(f"⚠️ Warning: Schema may not be valid JSON Schema (missing $schema)")
            print(f"✅ Schema syntax valid: {args.schema}")
        except Exception as e:
            print(f"❌ Error parsing schema: {str(e)}")
            return 1
        
        # Load and validate config
        try:
            config = load_json(args.config)
            print(f"✅ Configuration file parsed successfully: {args.config}")
            
            # Basic validation
            validate_config(config)
            print("✅ Configuration passes basic validation")
            
            # Schema validation
            schema_errors = validate_against_schema(config, schema)
            if schema_errors:
                print(f"❌ Schema validation failed with {len(schema_errors)} errors:")
                for i, error in enumerate(schema_errors, 1):
                    print(f"  {i}. {error}")
                return 1
            else:
                print("✅ Configuration passes schema validation")
            
            # Check referenced files
            check_referenced_files(config)
            
        except ConfigError as e:
            print(f"❌ Configuration error: {str(e)}")
            return 1
        
        # Check environment overrides
        env_overrides = find_env_overrides(args.env_prefix)
        if env_overrides:
            print("\n✅ Found environment overrides:")
            for key, value in env_overrides.items():
                print(f"   - {args.env_prefix}{key.upper()} = {value}")
        else:
            print("\n✅ No environment overrides found (using config file values)")
        
        # Show environment usage if requested
        if args.show_env:
            print_env_usage()
        
        print("\n✅ Validation complete. Configuration is valid.\n")
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
