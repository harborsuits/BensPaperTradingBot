#!/usr/bin/env python3
"""
Configuration Loader

This module provides utilities for loading and validating configuration files
for the BensBot trading system, with support for migration from legacy formats.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

from pydantic import ValidationError
import dotenv

from trading_bot.config.models import BotConfig, ConfigVersion

# Load environment variables from .env file if present
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration errors"""
    pass


def load_config(config_path: Union[str, Path]) -> BotConfig:
    """
    Load and validate configuration from a file path.
    
    Args:
        config_path: Path to the configuration file (JSON or YAML)
        
    Returns:
        Validated BotConfig instance
        
    Raises:
        ConfigurationError: If config cannot be loaded or is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    # Determine file type by extension
    file_extension = config_path.suffix.lower()
    
    try:
        # Load the raw configuration data
        if file_extension in ('.json', '.jsonc'):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif file_extension in ('.yaml', '.yml'):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ConfigurationError(f"Unsupported configuration file format: {file_extension}")
        
        # Check if this is a legacy config that needs migration
        if 'version' not in config_data:
            logger.warning("Legacy configuration detected, attempting migration...")
            config_data = migrate_legacy_config(config_data, config_path)
        
        # Validate configuration using Pydantic model
        try:
            config = BotConfig(**config_data)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ConfigurationError(f"Failed to parse configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}")


def save_config(config: BotConfig, output_path: Union[str, Path], format: str = 'json') -> None:
    """
    Save a BotConfig instance to a file.
    
    Args:
        config: BotConfig instance to save
        output_path: Path to save the configuration to
        format: Format to save as ('json' or 'yaml')
        
    Raises:
        ConfigurationError: If saving fails
    """
    output_path = Path(output_path)
    
    # Convert to dictionary
    config_dict = config.model_dump(exclude_none=True)
    
    try:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in specified format
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format.lower() == 'yaml':
            with open(output_path, 'w') as f:
                yaml.safe_dump(config_dict, f)
        else:
            raise ConfigurationError(f"Unsupported output format: {format}")
        
        logger.info(f"Configuration saved to {output_path}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {e}")


def migrate_legacy_config(legacy_config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """
    Migrate a legacy configuration to the new format.
    
    Args:
        legacy_config: Legacy configuration dictionary
        config_path: Path to the legacy configuration file
        
    Returns:
        Migrated configuration dictionary
    """
    # Detect legacy config type based on structure
    config_name = config_path.name
    parent_dir = config_path.parent.name
    
    # Start with a base structure for the new config
    new_config = {
        "version": ConfigVersion.V1_1,
        "environment": "development",
    }
    
    # Handle different legacy config types
    if "broker_config" in legacy_config:
        # Looks like a broker config
        new_config["broker_manager"] = _migrate_broker_config(legacy_config)
    elif "strategies" in legacy_config:
        # Looks like a strategy config
        new_config["strategy_manager"] = _migrate_strategy_config(legacy_config)
    elif "max_drawdown_pct" in legacy_config:
        # Looks like a risk management config
        new_config["risk_manager"] = _migrate_risk_config(legacy_config)
    elif "mongodb" in legacy_config and "redis" in legacy_config:
        # Looks like a persistence config
        new_config["persistence"] = legacy_config
    
    # Log the migration
    logger.info(f"Migrated legacy config: {config_path}")
    
    # Check if we need to add placeholders for required sections
    required_sections = ["persistence", "broker_manager", "risk_manager", "strategy_manager"]
    for section in required_sections:
        if section not in new_config:
            new_config[section] = _get_placeholder_config(section)
    
    return new_config


def _migrate_broker_config(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy broker configuration"""
    brokers = []
    asset_routing = []
    
    # Extract broker configurations
    if "brokers" in legacy_config:
        for broker in legacy_config["brokers"]:
            new_broker = {
                "id": broker.get("id", f"broker_{len(brokers)}"),
                "name": broker.get("name", broker.get("id", "Unknown Broker")),
                "type": broker.get("type", "unknown"),
                "enabled": broker.get("enabled", True),
                "sandbox_mode": broker.get("sandbox_mode", True),
                "timeout_seconds": broker.get("timeout_seconds", 30),
                "retry_attempts": broker.get("retry_attempts", 3),
                "credentials": {
                    "api_key": broker.get("api_key", "env:API_KEY"),
                    "api_secret": broker.get("api_secret"),
                    "account_id": broker.get("account_id"),
                    "additional_params": {}
                }
            }
            brokers.append(new_broker)
    
    # Extract asset routing rules
    if "asset_routing" in legacy_config:
        for rule in legacy_config["asset_routing"]:
            new_rule = {
                "asset_type": rule.get("asset_type", "stock"),
                "symbols": rule.get("symbols"),
                "market": rule.get("market"),
                "broker_id": rule.get("broker_id", brokers[0]["id"] if brokers else "default"),
                "priority": rule.get("priority", 1)
            }
            asset_routing.append(new_rule)
    
    return {
        "brokers": brokers,
        "asset_routing": asset_routing,
        "failover_enabled": legacy_config.get("failover_enabled", True),
        "metrics_enabled": legacy_config.get("metrics_enabled", True),
        "quote_cache_ttl_seconds": legacy_config.get("quote_cache_ttl_seconds", 5)
    }


def _migrate_strategy_config(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy strategy configuration"""
    strategies = []
    
    # Extract strategy configurations
    if "strategies" in legacy_config:
        for strategy_id, strategy in legacy_config["strategies"].items():
            new_strategy = {
                "id": strategy_id,
                "name": strategy.get("name", strategy_id),
                "type": strategy.get("type", "unknown"),
                "enabled": strategy.get("enabled", True),
                "assets": strategy.get("symbols", []),
                "parameters": strategy.get("parameters", {}),
                "schedule": strategy.get("schedule"),
                "risk_constraints": strategy.get("risk_constraints")
            }
            strategies.append(new_strategy)
    
    return {
        "strategies": strategies,
        "rotation_enabled": legacy_config.get("rotation_enabled", False),
        "rotation_interval_hours": legacy_config.get("rotation_interval_hours"),
        "concurrent_strategies_limit": legacy_config.get("concurrent_strategies_limit", 5)
    }


def _migrate_risk_config(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy risk management configuration"""
    return {
        "max_drawdown_pct": legacy_config.get("max_drawdown_pct", 5.0),
        "volatility_threshold": legacy_config.get("volatility_threshold", 2.5),
        "cooldown_minutes": legacy_config.get("cooldown_minutes", 60),
        "margin_call_threshold": legacy_config.get("margin_call_threshold", 0.25),
        "margin_warning_threshold": legacy_config.get("margin_warning_threshold", 0.35),
        "max_leverage": legacy_config.get("max_leverage", 2.0),
        "position_size_limit_pct": legacy_config.get("position_size_limit_pct", 5.0),
        "max_correlated_positions": legacy_config.get("max_correlated_positions", 3)
    }


def _get_placeholder_config(section: str) -> Dict[str, Any]:
    """Get placeholder configuration for a required section"""
    if section == "persistence":
        return {
            "mongodb": {
                "uri": "mongodb://localhost:27017",
                "database": "bensbot_trading",
                "max_pool_size": 20,
                "timeout_ms": 5000,
                "retry_writes": True,
                "retry_reads": True
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "timeout": 5.0,
                "decode_responses": True,
                "key_prefix": "bensbot:"
            },
            "recovery": {
                "recover_on_startup": True,
                "recover_open_orders": True,
                "recover_positions": True,
                "recover_pnl": True
            },
            "sync": {
                "periodic_sync_enabled": True,
                "sync_interval_seconds": 3600
            }
        }
    elif section == "broker_manager":
        return {
            "brokers": [
                {
                    "id": "default_broker",
                    "name": "Default Broker",
                    "type": "tradier",
                    "enabled": True,
                    "sandbox_mode": True,
                    "timeout_seconds": 30,
                    "retry_attempts": 3,
                    "credentials": {
                        "api_key": "env:API_KEY",
                        "account_id": "env:ACCOUNT_ID",
                        "additional_params": {}
                    }
                }
            ],
            "asset_routing": [],
            "failover_enabled": True,
            "metrics_enabled": True,
            "quote_cache_ttl_seconds": 5
        }
    elif section == "risk_manager":
        return {
            "max_drawdown_pct": 5.0,
            "volatility_threshold": 2.5,
            "cooldown_minutes": 60,
            "margin_call_threshold": 0.25,
            "margin_warning_threshold": 0.35,
            "max_leverage": 2.0,
            "position_size_limit_pct": 5.0,
            "max_correlated_positions": 3
        }
    elif section == "strategy_manager":
        return {
            "strategies": [],
            "rotation_enabled": False,
            "concurrent_strategies_limit": 5
        }
    else:
        return {}


def list_legacy_configs(base_dir: Union[str, Path]) -> List[Path]:
    """
    Find potential legacy configuration files.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of paths to potential legacy configuration files
    """
    base_dir = Path(base_dir)
    legacy_configs = []
    
    # Potential config file patterns
    patterns = [
        "**/*.json",
        "**/*.yaml",
        "**/*.yml",
        "**/config*.py"
    ]
    
    # Search for matching files
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file() and "config" in path.name.lower():
                legacy_configs.append(path)
    
    return legacy_configs


def create_migration_report(config_paths: List[Path]) -> str:
    """
    Create a report on legacy configurations found.
    
    Args:
        config_paths: List of configuration file paths
        
    Returns:
        Migration report as a string
    """
    report = "=== BensBot Configuration Migration Report ===\n\n"
    report += f"Found {len(config_paths)} potential configuration files:\n\n"
    
    for i, path in enumerate(config_paths, 1):
        report += f"{i}. {path}\n"
        
        # Try to determine config type
        config_type = "Unknown"
        try:
            if path.suffix.lower() in ('.json', '.jsonc'):
                with open(path, 'r') as f:
                    data = json.load(f)
            elif path.suffix.lower() in ('.yaml', '.yml'):
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                continue
                
            if "broker_config" in data or "brokers" in data:
                config_type = "Broker Configuration"
            elif "strategies" in data:
                config_type = "Strategy Configuration"
            elif "max_drawdown_pct" in data:
                config_type = "Risk Management Configuration"
            elif "mongodb" in data and "redis" in data:
                config_type = "Persistence Configuration"
        except:
            config_type = "Invalid or Unreadable"
            
        report += f"   Type: {config_type}\n"
        report += f"   Last Modified: {path.stat().st_mtime}\n\n"
    
    report += """
Migration Instructions:
1. Run the migration tool: python -m trading_bot.config.migrate_configs
2. This will create a unified config.yaml file with all settings
3. Review the generated config file and fill in any missing values
4. Update any scripts to use the new configuration loader
"""
    
    return report
