#!/usr/bin/env python
"""
Configuration Validator

This script validates the system configuration against its schema
to ensure all settings are correctly formatted and within valid ranges.
"""

import os
import sys
import json
import jsonschema
import argparse
from jsonschema import Draft7Validator

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(file_path):
    """Load JSON from file path"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {str(e)}")
        sys.exit(1)


def validate_config(config_path, schema_path):
    """Validate configuration against schema"""
    # Load schema and config
    schema = load_json(schema_path)
    config = load_json(config_path)
    
    # Create validator
    validator = Draft7Validator(schema)
    
    # Validate
    errors = list(validator.iter_errors(config))
    
    if errors:
        print(f"❌ Configuration validation failed with {len(errors)} errors:")
        for i, error in enumerate(errors, 1):
            print(f"\n--- Error {i} ---")
            print(f"Path: {' -> '.join([str(p) for p in error.path])}")
            print(f"Message: {error.message}")
            print(f"Schema path: {' -> '.join([str(p) for p in error.schema_path])}")
        return False
    else:
        print(f"✅ Configuration valid: {config_path}")
        return True


def check_additional_files(config):
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


def validate_environment():
    """Check for required environment variables"""
    env_vars = {
        "BENBOT_INITIAL_CAPITAL": "Override initial capital",
        "BENBOT_RISK_PER_TRADE": "Override risk per trade",
        "BENBOT_MAX_OPEN_POSITIONS": "Override max open positions",
        "BENBOT_LOG_LEVEL": "Override log level"
    }
    
    found_env_vars = []
    for var, description in env_vars.items():
        if var in os.environ:
            found_env_vars.append(f"{var}={os.environ[var]} ({description})")
    
    if found_env_vars:
        print("\n✅ Found environment overrides:")
        for var in found_env_vars:
            print(f"   - {var}")
    else:
        print("\n✅ No environment overrides found (using config file values)")


def main():
    parser = argparse.ArgumentParser(description='Validate BensBot configuration')
    parser.add_argument('--config', default='config/system_config.json', help='Path to configuration file')
    parser.add_argument('--schema', default='config/system_config.schema.json', help='Path to schema file')
    args = parser.parse_args()
    
    print("\n=== BensBot Configuration Validator ===\n")
    
    # Validate schema format
    try:
        schema = load_json(args.schema)
        # Simple check to see if it's a valid schema
        if '$schema' not in schema:
            print(f"⚠️ Warning: Schema may not be valid JSON Schema (missing $schema)")
        print(f"✅ Schema syntax valid: {args.schema}")
    except Exception as e:
        print(f"❌ Error parsing schema: {str(e)}")
        return 1
    
    # Validate config against schema
    if not validate_config(args.config, args.schema):
        return 1
    
    # Check referenced files
    config = load_json(args.config)
    check_additional_files(config)
    
    # Check environment variables
    validate_environment()
    
    print("\n✅ Validation complete. Configuration is valid.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
