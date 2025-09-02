#!/usr/bin/env python
"""
Configuration CLI

Command-line utility for working with BensBot configuration.
Allows checking, validating, and exploring configuration settings.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.core.simple_config import (
    load_config,
    apply_env_overrides,
    get_nested_value,
    ConfigError
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config_cli")


def setup_parser():
    """Set up argument parser"""
    parser = argparse.ArgumentParser(
        description="BensBot Configuration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_cli.py validate                          # Validate config
  python config_cli.py print                             # Print entire config
  python config_cli.py get trading_hours.start           # Get specific value
  python config_cli.py envs                              # Show env overrides
  python config_cli.py generate-env BENBOT_RISK_PER_TRADE=0.05 # Generate .env
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Validate config
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--config", default="config/system_config.json", help="Path to config file")
    
    # Print config
    print_parser = subparsers.add_parser("print", help="Print configuration")
    print_parser.add_argument("--config", default="config/system_config.json", help="Path to config file")
    print_parser.add_argument("--format", choices=["json", "flat"], default="json", help="Output format")
    
    # Get value
    get_parser = subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("path", help="Path to value (e.g. trading_hours.start)")
    get_parser.add_argument("--config", default="config/system_config.json", help="Path to config file")
    get_parser.add_argument("--default", help="Default value if path not found")
    
    # Show environment variables
    env_parser = subparsers.add_parser("envs", help="Show environment variable overrides")
    env_parser.add_argument("--prefix", default="BENBOT_", help="Environment variable prefix")
    
    # Generate .env file
    gen_env_parser = subparsers.add_parser("generate-env", help="Generate .env file")
    gen_env_parser.add_argument("--config", default="config/system_config.json", help="Path to config file")
    gen_env_parser.add_argument("--output", default=".env", help="Output file path")
    gen_env_parser.add_argument("vars", nargs="*", help="Variables to set (e.g. BENBOT_RISK_PER_TRADE=0.05)")
    
    return parser


def flatten_config(config, parent_key='', sep='.'):
    """Flatten nested dictionary into dot notation"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)


def cmd_validate(args):
    """Validate configuration file"""
    try:
        config = load_config(args.config)
        print(f"✅ Configuration is valid: {args.config}")
        return 0
    except ConfigError as e:
        print(f"❌ Configuration error: {str(e)}")
        return 1


def cmd_print(args):
    """Print configuration"""
    try:
        config = load_config(args.config)
        
        if args.format == "json":
            print(json.dumps(config, indent=2))
        else:  # flat
            flat_config = flatten_config(config)
            max_key_len = max(len(k) for k in flat_config.keys())
            
            for key, value in sorted(flat_config.items()):
                print(f"{key.ljust(max_key_len)} = {value}")
        
        return 0
    except ConfigError as e:
        print(f"❌ Configuration error: {str(e)}")
        return 1


def cmd_get(args):
    """Get configuration value"""
    try:
        config = load_config(args.config)
        value = get_nested_value(config, args.path, args.default)
        
        if value is None:
            print(f"Path not found: {args.path}")
            return 1
        
        # Pretty print values
        if isinstance(value, (dict, list)):
            print(json.dumps(value, indent=2))
        else:
            print(value)
            
        return 0
    except ConfigError as e:
        print(f"❌ Configuration error: {str(e)}")
        return 1


def cmd_envs(args):
    """Show environment variable overrides"""
    env_vars = {}
    
    # Find all environment variables with the prefix
    for key, value in os.environ.items():
        if key.startswith(args.prefix):
            env_vars[key] = value
    
    if env_vars:
        print(f"Environment variable overrides with prefix '{args.prefix}':")
        max_key_len = max(len(k) for k in env_vars.keys())
        
        for key, value in sorted(env_vars.items()):
            print(f"  {key.ljust(max_key_len)} = {value}")
    else:
        print(f"No environment variables found with prefix '{args.prefix}'")
    
    return 0


def cmd_generate_env(args):
    """Generate .env file"""
    try:
        # Parse variables
        env_vars = {}
        for var in args.vars:
            if "=" in var:
                key, value = var.split("=", 1)
                env_vars[key] = value
            else:
                print(f"Skipping invalid variable format: {var} (must be KEY=VALUE)")
        
        # Load existing .env file if it exists
        env_path = Path(args.output)
        existing_vars = {}
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing_vars[key] = value
        
        # Merge with new variables
        existing_vars.update(env_vars)
        
        # Write to file
        with open(env_path, 'w') as f:
            f.write("# BensBot environment variables\n")
            f.write("# Generated by config_cli.py\n\n")
            
            for key, value in sorted(existing_vars.items()):
                f.write(f"{key}={value}\n")
        
        print(f"✅ Environment file written to {args.output} with {len(existing_vars)} variables")
        return 0
    except Exception as e:
        print(f"❌ Error generating .env file: {str(e)}")
        return 1


def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == "validate":
        return cmd_validate(args)
    elif args.command == "print":
        return cmd_print(args)
    elif args.command == "get":
        return cmd_get(args)
    elif args.command == "envs":
        return cmd_envs(args)
    elif args.command == "generate-env":
        return cmd_generate_env(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
