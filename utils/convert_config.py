#!/usr/bin/env python3
"""
Config Migration Utility

This script converts old BensBot configuration formats to the new unified YAML format.
It supports:
- Converting old JSON config files to new YAML structure
- Merging multiple config files into a single unified format
- Reading environment variables and incorporating them into the config

Usage:
    python utils/convert_config.py --input old_config.json --output new_config.yaml
    python utils/convert_config.py --input old_config_dir/ --output new_config.yaml
    python utils/convert_config.py --input old_config.json --include-env --output new_config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import glob
import re
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime


# Mapping from old config keys to new config structure
# Format: "old_key": ("new_section", "new_key")
CONFIG_KEY_MAPPING = {
    # Legacy trading config
    "initial_balance": ("account", "initial_balance"),
    "risk_per_trade": ("risk", "max_risk_per_trade"),
    "max_trades": ("risk", "max_concurrent_trades"),
    "trailing_stop": ("risk", "use_trailing_stop"),
    "trailing_stop_pct": ("risk", "trailing_stop_percent"),
    "take_profit_pct": ("risk", "take_profit_percent"),
    "paper_trading": ("account", "paper_trading"),
    "log_level": ("logging", "level"),
    "log_file": ("logging", "file_path"),
    
    # Legacy broker config
    "api_key": ("brokers.credentials", "api_key"),
    "api_secret": ("brokers.credentials", "api_secret"),
    "account_number": ("brokers.credentials", "account_number"),
    "broker_api_url": ("brokers.settings", "api_url"),
    "tradier_account_id": ("brokers.credentials.tradier", "account_id"),
    "tradier_token": ("brokers.credentials.tradier", "token"),
    "alpaca_key_id": ("brokers.credentials.alpaca", "api_key"),
    "alpaca_secret_key": ("brokers.credentials.alpaca", "api_secret"),
    "alpaca_paper": ("brokers.settings.alpaca", "paper"),
    
    # Legacy strategy config
    "strategy_name": ("strategy", "name"),
    "strategy_class": ("strategy", "class"),
    "strategy_params": ("strategy", "parameters"),
    "symbols": ("data", "symbols"),
    "timeframe": ("data", "timeframe"),
    "strategy_mode": ("strategy", "mode"),
    
    # Legacy backtest config
    "start_date": ("backtest", "start_date"),
    "end_date": ("backtest", "end_date"),
    "data_source": ("data", "source"),
    "data_file": ("data", "file_path"),
    "commission": ("backtest", "commission_rate"),
    "slippage": ("backtest", "slippage_rate"),
}

# Environment variables mapping to config keys
# Format: "ENV_VAR": ("section", "key")
ENV_VAR_MAPPING = {
    "TRADIER_TOKEN": ("brokers.credentials.tradier", "token"),
    "TRADIER_ACCOUNT_ID": ("brokers.credentials.tradier", "account_id"),
    "ALPACA_API_KEY": ("brokers.credentials.alpaca", "api_key"),
    "ALPACA_SECRET_KEY": ("brokers.credentials.alpaca", "api_secret"),
    "ETRADE_CONSUMER_KEY": ("brokers.credentials.etrade", "consumer_key"),
    "ETRADE_CONSUMER_SECRET": ("brokers.credentials.etrade", "consumer_secret"),
    "API_KEY": ("brokers.credentials", "api_key"),
    "API_SECRET": ("brokers.credentials", "api_secret"),
    "MONGODB_URI": ("database", "mongodb_uri"),
    "REDIS_URI": ("database", "redis_uri"),
    "JWT_SECRET": ("security", "jwt_secret"),
    "DASHBOARD_ADMIN_PASSWORD": ("dashboard", "admin_password"),
    "LOG_LEVEL": ("logging", "level"),
}


def determine_file_type(file_path: str) -> str:
    """Determine the type of configuration file."""
    file_name = os.path.basename(file_path).lower()
    
    if 'broker' in file_name:
        return 'broker'
    elif 'strategy' in file_name:
        return 'strategy'
    elif 'backtest' in file_name:
        return 'backtest'
    elif 'log' in file_name:
        return 'logging'
    elif 'trading' in file_name:
        return 'trading'
    else:
        # Try to determine by content
        with open(file_path, 'r') as f:
            content = f.read()
            if any(term in content for term in ['api_key', 'broker', 'token', 'alpaca', 'tradier']):
                return 'broker'
            elif any(term in content for term in ['strategy', 'indicator', 'signal', 'entry']):
                return 'strategy'
            elif any(term in content for term in ['backtest', 'start_date', 'historical']):
                return 'backtest'
            elif any(term in content for term in ['log', 'console', 'debug']):
                return 'logging'
            else:
                return 'general'


def load_config_file(file_path: str) -> Dict[str, Any]:
    """Load a configuration file (JSON, YAML, or INI)."""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        elif ext == '.ini':
            # Simple INI parser (for basic use cases only)
            config = {}
            section = 'DEFAULT'
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith(';'):
                        continue
                    
                    # Check for section header
                    if line.startswith('[') and line.endswith(']'):
                        section = line[1:-1]
                        if section not in config:
                            config[section] = {}
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Try converting value to appropriate type
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif re.match(r'^-?\d+\.\d+$', value):
                            value = float(value)
                        
                        if section not in config:
                            config[section] = {}
                        config[section][key] = value
            
            return config
        elif ext == '.py':
            # Try to load Python variables from a module
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            config = {}
            for key in dir(module):
                if not key.startswith('__'):
                    value = getattr(module, key)
                    if not callable(value) and not key.startswith('_'):
                        config[key] = value
            
            return config
        else:
            print(f"Warning: Unsupported file extension {ext} for {file_path}. Trying as plain text.")
            # Try a basic key=value parser for unknown formats
            config = {}
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            
            return config
    except Exception as e:
        print(f"Error reading config file {file_path}: {e}")
        return {}


def set_nested_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dictionary structure using a dot-separated path."""
    if not path:
        return
    
    parts = path.split('.')
    current = config
    
    # Navigate to the deepest level
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            # If the current value is not a dict, convert it to one with a special key
            old_value = current[part]
            current[part] = {"_value": old_value}
        
        current = current[part]
    
    # Set the value at the deepest level
    current[parts[-1]] = value


def migrate_old_config(old_config: Dict[str, Any], config_type: str = None) -> Dict[str, Any]:
    """Migrate an old config structure to the new format."""
    new_config = {}
    
    # Handle nested dictionaries in old config
    def process_dict(d, prefix=""):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                process_dict(value, full_key)
            else:
                # Check if this is a known key to map
                if key in CONFIG_KEY_MAPPING:
                    new_section, new_key = CONFIG_KEY_MAPPING[key]
                    set_nested_value(new_config, f"{new_section}.{new_key}", value)
                else:
                    # Try to place it in an appropriate section based on config_type
                    if config_type == 'broker':
                        set_nested_value(new_config, f"brokers.settings.{key}", value)
                    elif config_type == 'strategy':
                        set_nested_value(new_config, f"strategy.parameters.{key}", value)
                    elif config_type == 'backtest':
                        set_nested_value(new_config, f"backtest.{key}", value)
                    elif config_type == 'logging':
                        set_nested_value(new_config, f"logging.{key}", value)
                    else:
                        # Just add it at the top level if we can't determine a better place
                        new_config[key] = value
    
    # Process the configuration
    process_dict(old_config)
    
    # Add metadata
    new_config['metadata'] = {
        'migrated_from': 'legacy_config',
        'migration_date': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return new_config


def get_env_vars() -> Dict[str, Any]:
    """Get relevant environment variables for the configuration."""
    env_config = {}
    
    for env_var, (section, key) in ENV_VAR_MAPPING.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Try to convert value to appropriate type
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)
            
            set_nested_value(env_config, f"{section}.{key}", value)
    
    return env_config


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple configurations into one."""
    if not configs:
        return {}
    
    if len(configs) == 1:
        return configs[0]
    
    result = {}
    
    def deep_merge(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                deep_merge(target[key], value)
            else:
                target[key] = value
    
    for config in configs:
        deep_merge(result, config)
    
    return result


def validate_new_config(config: Dict[str, Any]) -> List[str]:
    """Perform basic validation on the new config format."""
    warnings = []
    
    # Check for required sections
    required_sections = ['brokers', 'strategy', 'risk', 'logging']
    for section in required_sections:
        if section not in config:
            warnings.append(f"Missing required section: {section}")
    
    # Check broker configuration
    if 'brokers' in config:
        if 'enabled' not in config['brokers']:
            warnings.append("Missing 'enabled' list in brokers section")
        
        if 'credentials' in config['brokers']:
            # Check credentials for enabled brokers
            if 'enabled' in config['brokers']:
                for broker in config['brokers']['enabled']:
                    if (broker not in config['brokers']['credentials'] and 
                        f"{broker}" not in config['brokers']['credentials']):
                        warnings.append(f"Credentials missing for enabled broker: {broker}")
    
    # Check strategy configuration
    if 'strategy' in config:
        if 'name' not in config['strategy'] and 'class' not in config['strategy']:
            warnings.append("Strategy section should have either 'name' or 'class'")
    
    # Check data configuration
    if 'data' in config and 'symbols' not in config['data']:
        warnings.append("Missing 'symbols' in data section")
    
    return warnings


def main():
    parser = argparse.ArgumentParser(description='Convert old BensBot config to new format')
    parser.add_argument('--input', '-i', required=True, help='Input config file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output YAML file')
    parser.add_argument('--include-env', '-e', action='store_true', 
                        help='Include environment variables in the output config')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite output file if it exists')
    args = parser.parse_args()
    
    # Check if output file exists
    if os.path.exists(args.output) and not args.force:
        print(f"Error: Output file {args.output} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    configs = []
    config_types = set()
    
    # Check if input is a directory or file
    if os.path.isdir(args.input):
        # Process all config files in the directory
        config_files = glob.glob(os.path.join(args.input, '*.json'))
        config_files += glob.glob(os.path.join(args.input, '*.yaml'))
        config_files += glob.glob(os.path.join(args.input, '*.yml'))
        config_files += glob.glob(os.path.join(args.input, '*.ini'))
        config_files += glob.glob(os.path.join(args.input, '*config*.py'))
        
        for file_path in config_files:
            print(f"Processing: {file_path}")
            config_type = determine_file_type(file_path)
            config_types.add(config_type)
            old_config = load_config_file(file_path)
            new_config = migrate_old_config(old_config, config_type)
            configs.append(new_config)
    else:
        # Process a single file
        print(f"Processing: {args.input}")
        config_type = determine_file_type(args.input)
        config_types.add(config_type)
        old_config = load_config_file(args.input)
        new_config = migrate_old_config(old_config, config_type)
        configs.append(new_config)
    
    # Include environment variables if requested
    if args.include_env:
        print("Including environment variables")
        env_config = get_env_vars()
        if env_config:
            configs.append(env_config)
    
    # Merge all configs
    merged_config = merge_configs(configs)
    
    # Add broker enablement if broker config is present
    if 'brokers' in merged_config and 'credentials' in merged_config['brokers']:
        if 'enabled' not in merged_config['brokers']:
            # By default, enable all brokers with credentials
            merged_config['brokers']['enabled'] = list(merged_config['brokers']['credentials'].keys())
    
    # Validate the merged config
    warnings = validate_new_config(merged_config)
    for warning in warnings:
        print(f"Warning: {warning}")
    
    # Write the merged config to the output file
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nMigration Summary:")
    print(f"- Processed configuration types: {', '.join(config_types)}")
    print(f"- Merged {len(configs)} configurations")
    print(f"- Found {len(warnings)} potential issues")
    print(f"- New config written to: {args.output}")
    
    if warnings:
        print("\nPlease review the warnings above and manually verify the output config.")
    else:
        print("\nMigration completed successfully!")
    
    # Provide helpful next steps
    print("\nNext Steps:")
    print("1. Review the generated YAML file to ensure all settings were correctly migrated")
    print("2. Check for any missing or incorrect values")
    print("3. Update your deployment scripts to use the new unified configuration")
    print("4. Run 'python run_bot.py --config your_new_config.yaml' to use the new setup")


if __name__ == "__main__":
    main()
