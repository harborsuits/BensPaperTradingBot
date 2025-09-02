#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert legacy JSON or Python dict configs to new YAML typed settings.

This script helps migrate older configuration formats to the new
Pydantic-based typed settings system, preserving existing values
and adding appropriate structure for the new format.

Usage:
    python migrate_config.py legacy_config.json [output_config.yaml]
    python migrate_config.py legacy_config.py [output_config.yaml]

If output_config.yaml is not specified, it defaults to "config.yaml"
in the current directory.
"""

import argparse
import json
import pathlib
import importlib.util
import yaml
import sys
import os
from typing import Dict, Any


def load_legacy(path: pathlib.Path) -> dict:
    """
    Load a legacy configuration file (JSON or Python).
    
    Args:
        path: Path to the legacy config file
        
    Returns:
        Dict containing the configuration
        
    Raises:
        ValueError: If the file format is not supported
    """
    if path.suffix == ".json":
        return json.loads(path.read_text())
    
    elif path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("cfg", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load Python module: {path}")
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Look for common config variable names
        for var_name in ["CONFIG", "CONFIG", "config", "settings", "SETTINGS"]:
            if hasattr(mod, var_name):
                return getattr(mod, var_name)
                
        raise ValueError(f"No CONFIG or SETTINGS variable found in {path}")
    
    raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .py")


def transform_config(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform legacy config format to the new typed settings structure.
    
    Args:
        legacy_config: The legacy configuration dictionary
        
    Returns:
        Dict containing the transformed configuration
    """
    # Start with an empty config in the new format
    new_config: Dict[str, Any] = {
        "broker": {},
        "risk": {},
        "data": {},
        "notifications": {},
        "api": {},
        "backtest": {},
        "logging": {},
        "ui": {}
    }
    
    # Map legacy keys to new structure
    # Broker settings
    if "broker" in legacy_config:
        if isinstance(legacy_config["broker"], str):
            new_config["broker"]["name"] = legacy_config["broker"]
        elif isinstance(legacy_config["broker"], dict):
            new_config["broker"].update(legacy_config["broker"])
    
    if "api_key" in legacy_config:
        new_config["broker"]["api_key"] = legacy_config["api_key"]
    
    if "account_id" in legacy_config:
        new_config["broker"]["account_id"] = legacy_config["account_id"]
    
    if "sandbox" in legacy_config:
        new_config["broker"]["sandbox"] = legacy_config["sandbox"]
    
    # Risk settings
    risk_keys = [
        "max_position_pct", "max_risk_pct", "max_portfolio_risk",
        "max_drawdown", "max_open_trades", "max_sector_allocation",
        "portfolio_stop_loss_pct", "portfolio_take_profit_pct"
    ]
    
    for key in risk_keys:
        if key in legacy_config:
            new_config["risk"][key] = legacy_config[key]
    
    # If there's a nested risk dict, use that too
    if "risk" in legacy_config and isinstance(legacy_config["risk"], dict):
        for key, value in legacy_config["risk"].items():
            new_config["risk"][key] = value
    
    # API settings
    api_keys = ["host", "port", "debug", "cors_origins", "rate_limit_requests"]
    for key in api_keys:
        if key in legacy_config:
            new_config["api"][key] = legacy_config[key]
    
    # If there's a nested api dict, use that too
    if "api" in legacy_config and isinstance(legacy_config["api"], dict):
        for key, value in legacy_config["api"].items():
            new_config["api"][key] = value
    
    # Handle API keys dictionary
    api_keys_dict = {}
    
    # Common API key locations in legacy configs
    api_providers = [
        "alpha_vantage", "finnhub", "tradier", "alpaca", 
        "marketaux", "newsdata", "gnews", "mediastack", "currents", "nytimes",
        "huggingface", "openai", "claude", "mistral", "cohere", "gemini"
    ]
    
    for provider in api_providers:
        key_name = f"{provider}_api_key"
        if key_name in legacy_config:
            api_keys_dict[provider] = legacy_config[key_name]
        elif provider in legacy_config:
            api_keys_dict[provider] = legacy_config[provider]
    
    if api_keys_dict:
        new_config["api"]["api_keys"] = api_keys_dict
    
    # Notification settings
    if "notifications" in legacy_config and isinstance(legacy_config["notifications"], dict):
        new_config["notifications"].update(legacy_config["notifications"])
    
    # Common notification keys that might be at the root
    notification_keys = ["telegram_token", "telegram_chat_id", "slack_webhook", "enable_notifications"]
    for key in notification_keys:
        if key in legacy_config:
            new_config["notifications"][key] = legacy_config[key]
    
    # Backtest settings
    if "backtest" in legacy_config and isinstance(legacy_config["backtest"], dict):
        new_config["backtest"].update(legacy_config["backtest"])
    
    backtest_keys = [
        "initial_capital", "default_symbols", "default_start_date", 
        "default_end_date", "commission_per_trade", "slippage_pct"
    ]
    
    for key in backtest_keys:
        if key in legacy_config:
            new_config["backtest"][key] = legacy_config[key]
    
    # Data settings
    if "data" in legacy_config and isinstance(legacy_config["data"], dict):
        new_config["data"].update(legacy_config["data"])
    
    data_keys = ["provider", "use_websocket", "cache_expiry_seconds", "historical_source"]
    for key in data_keys:
        if key in legacy_config:
            new_config["data"][key] = legacy_config[key]
    
    # Clean up empty sections
    for section in list(new_config.keys()):
        if not new_config[section]:
            del new_config[section]
    
    return new_config


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert legacy config files to new typed settings YAML format"
    )
    parser.add_argument("infile", help="Legacy configuration file (.json or .py)")
    parser.add_argument("outfile", nargs="?", default="config.yaml", 
                       help="Output YAML file (default: config.yaml)")
    
    args = parser.parse_args()
    
    try:
        inpath = pathlib.Path(args.infile)
        outpath = pathlib.Path(args.outfile)
        
        if not inpath.exists():
            print(f"Error: Input file {inpath} does not exist", file=sys.stderr)
            return 1
        
        # Load legacy config
        print(f"Loading legacy config from {inpath}...")
        legacy_config = load_legacy(inpath)
        
        # Transform to new format
        print("Transforming to new typed settings format...")
        new_config = transform_config(legacy_config)
        
        # Write to YAML file
        outpath.write_text(yaml.dump(new_config, sort_keys=False))
        print(f"✅ Successfully wrote new config to {outpath}")
        
        # Give user a hint about verification
        print("\nNext steps:")
        print("1. Review the generated config file to ensure all settings were properly migrated")
        print("2. Consider moving sensitive data like API keys to environment variables")
        print("3. Test the configuration with your application")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
