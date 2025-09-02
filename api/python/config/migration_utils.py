"""
Configuration Migration Utilities

This module provides utilities to help migrate from legacy configuration systems
to the new typed_settings based configuration system.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, Union, Callable, Type, TypeVar

# Import the typed settings module
from trading_bot.config.typed_settings import (
    TradingBotSettings, 
    BrokerSettings,
    RiskSettings,
    DataSourceSettings,
    NotificationSettings,
    OrchestratorSettings,
    BacktestSettings,
    LoggingSettings,
    UISettings,
    load_config as typed_load_config
)

logger = logging.getLogger(__name__)

# Type for settings models
T = TypeVar('T')

def migrate_config(
    legacy_config: Dict[str, Any], 
    config_type: Type[T] = TradingBotSettings
) -> T:
    """
    Migrate a legacy configuration dictionary to the new typed settings system.
    
    Args:
        legacy_config: Legacy configuration dictionary
        config_type: The typed settings class to migrate to
        
    Returns:
        An instance of the specified config_type
    """
    # Try to convert the legacy config to the new format
    try:
        # For the full TradingBotSettings, we need to restructure the legacy config
        if config_type is TradingBotSettings:
            return _migrate_full_config(legacy_config)
        
        # For component configs, just validate against the appropriate model
        return config_type(**legacy_config)
        
    except Exception as e:
        logger.error(f"Error migrating legacy config: {str(e)}")
        logger.error("Falling back to default values where possible")
        
        # Fall back to defaults with as much of the legacy config as possible
        if config_type is TradingBotSettings:
            # Extract any valid broker settings as a minimum
            broker_config = legacy_config.get("broker", {})
            if "api_key" in broker_config and "account_id" in broker_config:
                return TradingBotSettings(broker=BrokerSettings(**broker_config))
        
        # Return defaults for the specified model
        return config_type()

def _migrate_full_config(legacy_config: Dict[str, Any]) -> TradingBotSettings:
    """
    Migrate a full legacy configuration to TradingBotSettings.
    
    Args:
        legacy_config: Legacy configuration dictionary
        
    Returns:
        TradingBotSettings instance
    """
    # Initialize new config structure
    new_config = {}
    
    # Map legacy broker config
    if "broker" in legacy_config:
        new_config["broker"] = legacy_config["broker"]
    elif "tradier" in legacy_config:
        # Some configs might have broker settings under "tradier" key
        new_config["broker"] = {
            "name": "tradier",
            "api_key": legacy_config["tradier"].get("api_key", ""),
            "account_id": legacy_config["tradier"].get("account_id", ""),
            "sandbox": legacy_config["tradier"].get("sandbox", True),
        }
    
    # Map legacy risk config
    if "risk" in legacy_config:
        new_config["risk"] = legacy_config["risk"]
    elif "risk_management" in legacy_config:
        # Some configs might have risk settings under "risk_management" key
        risk_config = legacy_config["risk_management"]
        new_config["risk"] = {
            "max_position_pct": risk_config.get("max_position_size", 0.05),
            "max_risk_pct": risk_config.get("max_risk_per_trade", 0.01),
            "max_portfolio_risk": risk_config.get("max_portfolio_risk", 0.20),
            "max_correlated_positions": risk_config.get("max_correlated", 3),
            "max_sector_allocation": risk_config.get("max_sector", 0.30),
            "max_open_trades": risk_config.get("max_trades", 5)
        }
    
    # Map legacy data source config
    if "data" in legacy_config:
        new_config["data"] = legacy_config["data"]
    elif "market_data" in legacy_config:
        data_config = legacy_config["market_data"]
        new_config["data"] = {
            "provider": data_config.get("provider", "tradier"),
            "api_keys": data_config.get("api_keys", {})
        }
    
    # Map legacy notification config
    if "notifications" in legacy_config:
        new_config["notifications"] = legacy_config["notifications"]
    elif "notification" in legacy_config:
        new_config["notifications"] = legacy_config["notification"]
    
    # Map legacy orchestrator config
    if "orchestrator" in legacy_config:
        new_config["orchestrator"] = legacy_config["orchestrator"]
    elif "trading_hours" in legacy_config:
        hours_config = legacy_config["trading_hours"]
        new_config["orchestrator"] = {
            "trading_hours_only": True,
            "market_hours_start": hours_config.get("start", "09:30"),
            "market_hours_end": hours_config.get("end", "16:00"),
            "timezone": hours_config.get("timezone", "America/New_York")
        }
    
    # Map legacy backtest config
    if "backtest" in legacy_config:
        new_config["backtest"] = legacy_config["backtest"]
    elif "backtesting" in legacy_config:
        new_config["backtest"] = legacy_config["backtesting"]
    
    # Map legacy logging config
    if "logging" in legacy_config:
        new_config["logging"] = legacy_config["logging"]
    elif "log" in legacy_config:
        log_config = legacy_config["log"]
        new_config["logging"] = {
            "level": log_config.get("level", "INFO"),
            "file_path": log_config.get("file", "./logs/trading_bot.log")
        }
    
    # Map legacy UI config
    if "ui" in legacy_config:
        new_config["ui"] = legacy_config["ui"]
    
    # Preserve environment and version settings
    new_config["environment"] = legacy_config.get("environment", "development")
    new_config["version"] = legacy_config.get("version", "1.0.0")
    
    # Create the validated settings object
    return TradingBotSettings(**new_config)

def get_config_from_legacy_path(
    legacy_path: Optional[str] = None,
    fallback_path: str = "./trading_bot/config/config.yaml"
) -> TradingBotSettings:
    """
    Load configuration from a legacy path, falling back to the new config path.
    
    Args:
        legacy_path: Legacy configuration file path
        fallback_path: Fallback configuration file path
        
    Returns:
        TradingBotSettings instance
    """
    # Try to load from the new typed settings first if no legacy path provided
    if not legacy_path:
        try:
            return typed_load_config(fallback_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {fallback_path}: {str(e)}")
            logger.warning("Will try to load from environment variables")
            return typed_load_config()
    
    # Try to load the legacy config
    try:
        legacy_config = {}
        
        # Load based on file extension
        if legacy_path.endswith(('.yaml', '.yml')):
            import yaml
            with open(legacy_path, 'r') as f:
                legacy_config = yaml.safe_load(f)
        elif legacy_path.endswith('.json'):
            with open(legacy_path, 'r') as f:
                legacy_config = json.load(f)
        else:
            # Try JSON first, then YAML
            try:
                with open(legacy_path, 'r') as f:
                    legacy_config = json.load(f)
            except json.JSONDecodeError:
                import yaml
                with open(legacy_path, 'r') as f:
                    legacy_config = yaml.safe_load(f)
        
        # Migrate the legacy config
        return migrate_config(legacy_config)
    
    except Exception as e:
        logger.error(f"Failed to load legacy config from {legacy_path}: {str(e)}")
        
        # Fall back to the new config
        try:
            return typed_load_config(fallback_path)
        except Exception:
            # Last resort: try environment variables
            logger.warning("Falling back to environment variables and defaults")
            return typed_load_config()

def migrate_module_config(module_name: str) -> None:
    """
    Migrate a module's configuration to use the typed settings system.
    
    This function should be called at the beginning of a module's execution
    to ensure it uses the new typed settings system.
    
    Args:
        module_name: Name of the module being migrated
    
    Example:
        if __name__ == "__main__":
            from trading_bot.config.migration_utils import migrate_module_config
            settings = migrate_module_config("my_module")
            # Use settings instead of the old config
    """
    logger.info(f"Migrating config for module: {module_name}")
    
    # Try to find the legacy config path from environment variables
    legacy_path = os.environ.get(f"{module_name.upper()}_CONFIG")
    
    # Load the config with proper migration
    settings = get_config_from_legacy_path(legacy_path)
    
    logger.info(f"Successfully migrated config for {module_name}")
    return settings

# Decorator to migrate a function that takes a config_path parameter
def with_migrated_config(func: Callable):
    """
    Decorator to migrate a function that uses a legacy config system.
    
    Example:
        @with_migrated_config
        def my_function(config_path=None, **kwargs):
            # The config_path will be automatically migrated
            # and passed to the function as a TradingBotSettings instance
    """
    def wrapper(config_path=None, **kwargs):
        settings = get_config_from_legacy_path(config_path)
        return func(settings, **kwargs)
    
    return wrapper
