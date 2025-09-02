#!/usr/bin/env python3
"""
Configuration Migration Utility

This script helps migrate legacy configuration files to the new unified format.
It scans for legacy config files, migrates them, and creates a unified config.
"""

import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from trading_bot.config.loader import (
    load_config,
    save_config,
    list_legacy_configs,
    create_migration_report,
    migrate_legacy_config,
    BotConfig
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("config_migration")


def find_configs(base_dir: Path) -> Dict[str, Path]:
    """
    Find legacy configuration files and categorize them.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        Dictionary of config types to paths
    """
    config_types = {
        "broker": None,
        "strategy": None,
        "risk": None,
        "persistence": None
    }
    
    # Find all potential config files
    config_paths = list_legacy_configs(base_dir)
    
    # Try to categorize each file
    for path in config_paths:
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
                config_types["broker"] = path
            elif "strategies" in data:
                config_types["strategy"] = path
            elif "max_drawdown_pct" in data:
                config_types["risk"] = path
            elif "mongodb" in data and "redis" in data:
                config_types["persistence"] = path
        except Exception as e:
            logger.warning(f"Could not process {path}: {e}")
    
    return config_types


def build_unified_config(config_paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    Build a unified configuration from multiple legacy configs.
    
    Args:
        config_paths: Dictionary of config types to paths
        
    Returns:
        Unified configuration dictionary
    """
    unified_config = {
        "version": "1.1",
        "environment": "development"
    }
    
    # Process each config type
    for config_type, path in config_paths.items():
        if path is None:
            continue
            
        try:
            if path.suffix.lower() in ('.json', '.jsonc'):
                with open(path, 'r') as f:
                    data = json.load(f)
            elif path.suffix.lower() in ('.yaml', '.yml'):
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                continue
                
            migrated_data = migrate_legacy_config(data, path)
            
            # Extract the relevant section
            if config_type == "broker" and "broker_manager" in migrated_data:
                unified_config["broker_manager"] = migrated_data["broker_manager"]
            elif config_type == "strategy" and "strategy_manager" in migrated_data:
                unified_config["strategy_manager"] = migrated_data["strategy_manager"]
            elif config_type == "risk" and "risk_manager" in migrated_data:
                unified_config["risk_manager"] = migrated_data["risk_manager"]
            elif config_type == "persistence" and "persistence" in migrated_data:
                unified_config["persistence"] = migrated_data["persistence"]
        except Exception as e:
            logger.warning(f"Could not migrate {path}: {e}")
    
    # Add placeholder sections for missing required sections
    if "persistence" not in unified_config:
        unified_config["persistence"] = {
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
    
    if "broker_manager" not in unified_config:
        unified_config["broker_manager"] = {
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
    
    if "risk_manager" not in unified_config:
        unified_config["risk_manager"] = {
            "max_drawdown_pct": 5.0,
            "volatility_threshold": 2.5,
            "cooldown_minutes": 60,
            "margin_call_threshold": 0.25,
            "margin_warning_threshold": 0.35,
            "max_leverage": 2.0,
            "position_size_limit_pct": 5.0,
            "max_correlated_positions": 3
        }
    
    if "strategy_manager" not in unified_config:
        unified_config["strategy_manager"] = {
            "strategies": [],
            "rotation_enabled": False,
            "concurrent_strategies_limit": 5
        }
    
    return unified_config


def main():
    """Main entry point for the migration utility"""
    parser = argparse.ArgumentParser(description="BensBot Configuration Migration Utility")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory to search for config files")
    parser.add_argument("--output", type=str, default="config/config.yaml", help="Output path for unified config")
    parser.add_argument("--format", type=str, choices=["json", "yaml"], default="yaml", help="Output format")
    parser.add_argument("--report", action="store_true", help="Generate a migration report only")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_path = Path(args.output)
    
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Find legacy configs
    legacy_configs = list_legacy_configs(base_dir)
    
    if args.report:
        # Generate and print migration report
        report = create_migration_report(legacy_configs)
        print(report)
        
        # Save report to file
        report_path = base_dir / "migration_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Migration report saved to {report_path}")
        
        sys.exit(0)
    
    # Categorize found configs
    config_types = find_configs(base_dir)
    
    # Log found configs
    logger.info("Found the following configuration files:")
    for config_type, path in config_types.items():
        if path:
            logger.info(f"- {config_type}: {path}")
        else:
            logger.warning(f"- {config_type}: Not found")
    
    # Build unified config
    unified_config = build_unified_config(config_types)
    
    # Validate using Pydantic model
    try:
        config = BotConfig(**unified_config)
        logger.info("Unified configuration validated successfully")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.info("Saving the configuration anyway, but manual fixes will be required")
    
    # Save unified config
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in specified format
        if args.format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(unified_config, f, indent=2)
        else:  # yaml
            with open(output_path, 'w') as f:
                yaml.safe_dump(unified_config, f)
        
        logger.info(f"Unified configuration saved to {output_path}")
        
        # Print next steps
        print("\nMigration completed successfully!")
        print(f"Unified configuration saved to {output_path}")
        print("\nNext steps:")
        print("1. Review the generated configuration file and fill in any missing values")
        print("2. Update any scripts to use the new configuration loader:")
        print("   from trading_bot.config.loader import load_config")
        print("   config = load_config('config/config.yaml')")
        
    except Exception as e:
        logger.error(f"Failed to save unified configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
