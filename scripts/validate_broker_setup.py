#!/usr/bin/env python3
"""
Broker Configuration Validator

A simplified script to validate broker configuration and credentials.
This script is designed to run independently without requiring the full package structure.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import credential manager (if available)
try:
    from trading_bot.core.credential_manager import CredentialManager, is_dev_environment
    CREDENTIAL_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIAL_MANAGER_AVAILABLE = False
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("broker_validator")

class ValidationError(Exception):
    """Exception raised when validation fails"""
    pass

def load_json_file(file_path):
    """Load JSON from file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise ValidationError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}")
        raise ValidationError(f"Invalid JSON in {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise ValidationError(f"Error loading {file_path}: {str(e)}")

def validate_broker_config(broker_config):
    """Validate broker configuration structure"""
    errors = []
    
    # Check required fields
    if "default_broker" not in broker_config:
        errors.append("Missing required field: default_broker")
    
    if "brokers" not in broker_config or not isinstance(broker_config["brokers"], dict):
        errors.append("Missing or invalid brokers configuration")
        return errors
    
    # Check if default broker exists
    if "default_broker" in broker_config:
        default_broker = broker_config["default_broker"]
        if default_broker not in broker_config["brokers"]:
            errors.append(f"Default broker '{default_broker}' not found in brokers configuration")
    
    # Check broker configurations
    for broker_id, broker_conf in broker_config["brokers"].items():
        # Check required broker fields
        for field in ["broker_type", "enabled"]:
            if field not in broker_conf:
                errors.append(f"Broker {broker_id} missing required field: {field}")
        
        # Check broker type
        if "broker_type" in broker_conf:
            broker_type = broker_conf["broker_type"]
            if broker_type not in ["alpaca", "interactive_brokers", "tradier", "paper"]:
                errors.append(f"Broker {broker_id} has invalid broker_type: {broker_type}")
    
    return errors

def validate_credentials(broker_id, broker_config, credential_path):
    """Check if credentials exist and are valid for a broker"""
    broker_type = broker_config.get("broker_type")
    
    # Skip disabled brokers and paper broker (which needs no credentials)
    if not broker_config.get("enabled", False) or broker_type == "paper":
        return []
    
    errors = []
    
    # Use credential manager if available
    if CREDENTIAL_MANAGER_AVAILABLE:
        # Initialize credential manager
        credential_manager = CredentialManager(
            credential_dir=credential_path,
            use_mock_in_dev=True,  # Use mock in dev for validation
            dev_mode=is_dev_environment()
        )
        
        # Get required fields based on broker type
        required_fields = []
        if broker_type == "alpaca":
            required_fields = ["api_key", "api_secret"]
        elif broker_type == "tradier":
            required_fields = ["api_key", "account_id"]
        elif broker_type == "interactive_brokers":
            required_fields = ["tws_host", "tws_port"]
            
        # Try to get credentials (allow missing for validation)
        try:
            credentials = credential_manager.get_broker_credentials(
                broker_id, broker_type, required_fields, allow_missing=True
            )
            
            # Check credential source
            source = credential_manager.get_credential_source(broker_id)
            
            # Validate required fields
            for field in required_fields:
                if field not in credentials:
                    errors.append(f"{broker_type.title()} broker {broker_id} missing required credential: {field}")
                    
            # Warn if using mock credentials
            if source == "mock":
                print(f"  ⚠️ Using mock credentials for {broker_id} (not for production use)")
                
        except Exception as e:
            errors.append(f"Error validating credentials for {broker_id}: {str(e)}")
            
    else:
        # Fall back to file-based validation if credential manager not available
        credential_file = os.path.join(credential_path, f"{broker_id}.json")
        
        # Check if credential file exists
        if not os.path.exists(credential_file):
            errors.append(f"Credential file not found for {broker_id}: {credential_file}")
            return errors
        
        # Load and check credentials
        try:
            credentials = load_json_file(credential_file)
            
            # Alpaca broker validation
            if broker_type == "alpaca":
                if not credentials.get("api_key"):
                    errors.append(f"Alpaca broker {broker_id} missing required credential: api_key")
                if not credentials.get("api_secret"):
                    errors.append(f"Alpaca broker {broker_id} missing required credential: api_secret")
            
            # Tradier broker validation
            elif broker_type == "tradier":
                if not credentials.get("api_key"):
                    errors.append(f"Tradier broker {broker_id} missing required credential: api_key")
                if not credentials.get("account_id"):
                    errors.append(f"Tradier broker {broker_id} missing required credential: account_id")
            
            # Interactive Brokers validation
            elif broker_type == "interactive_brokers":
                if not credentials.get("tws_host"):
                    errors.append(f"IB broker {broker_id} missing required credential: tws_host")
                if not credentials.get("tws_port"):
                    errors.append(f"IB broker {broker_id} missing required credential: tws_port")
        
        except ValidationError:
            errors.append(f"Failed to load credentials for {broker_id}")
    
    return errors

def check_env_overrides():
    """Check for broker-related environment variables"""
    broker_vars = []
    for key, value in os.environ.items():
        if key.startswith("BENBOT_BROKERS_") or key == "BENBOT_DEFAULT_BROKER":
            broker_vars.append((key, value))
    return broker_vars

def validate_all_brokers(config_path, credential_path=None, skip_credentials=False):
    """Validate broker configuration and credentials"""
    try:
        # Load broker configuration
        print(f"\nLoading broker configuration from: {config_path}")
        broker_config = load_json_file(config_path)
        
        # Get credential path from config if not provided
        if credential_path is None:
            credential_path = broker_config.get("credential_path", "config/credentials")
        
        # Check if credential directory exists
        if not os.path.isdir(credential_path) and not skip_credentials:
            os.makedirs(credential_path, exist_ok=True)
            print(f"\nCreated credential directory: {credential_path}")
        
        # Validate broker configuration
        print("\nValidating broker configuration...")
        config_errors = validate_broker_config(broker_config)
        
        if config_errors:
            print("\n❌ Broker configuration has errors:")
            for error in config_errors:
                print(f"  - {error}")
            return False
        
        print("✅ Broker configuration structure is valid")
        
        # Check for environment variable overrides
        env_overrides = check_env_overrides()
        if env_overrides:
            print("\nBroker environment variable overrides found:")
            for var, value in env_overrides:
                print(f"  {var}={value}")
        
        # Skip credential validation if requested
        if skip_credentials:
            print("\n⚠️ Skipping credential validation as requested")
            return len(config_errors) == 0
        
        # Validate credentials for each broker
        print("\nValidating broker credentials...")
        
        all_valid = True
        brokers = broker_config.get("brokers", {})
        enabled_brokers = [b_id for b_id, b_conf in brokers.items() if b_conf.get("enabled", False)]
        
        print(f"Found {len(enabled_brokers)} enabled broker(s) out of {len(brokers)} total")
        
        for broker_id, broker_conf in brokers.items():
            broker_type = broker_conf.get("broker_type", "unknown")
            enabled = broker_conf.get("enabled", False)
            
            print(f"\n{'✓' if enabled else '✗'} Broker: {broker_id} ({broker_type})")
            
            if not enabled:
                print("  ℹ️ Broker is disabled, skipping credential validation")
                continue
            
            if broker_type == "paper":
                print("  ✅ Paper broker (no credentials required)")
                continue
            
            # Validate credentials
            cred_errors = validate_credentials(broker_id, broker_conf, credential_path)
            
            if cred_errors:
                print("  ❌ Credential validation failed:")
                for error in cred_errors:
                    print(f"    - {error}")
                all_valid = False
            else:
                print("  ✅ Credentials validation passed")
        
        # Print summary
        print("\n=== Broker Validation Summary ===")
        if all_valid and len(config_errors) == 0:
            print("✅ All broker validations passed")
        else:
            print("❌ Some broker validations failed")
        
        return all_valid and len(config_errors) == 0
        
    except ValidationError as e:
        print(f"\n❌ Validation error: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BensBot Broker Validator")
    parser.add_argument("--config", default="config/broker_config.json", help="Path to broker configuration file")
    parser.add_argument("--credentials", default=None, help="Path to credentials directory (default: from config)")
    parser.add_argument("--skip-credentials", action="store_true", help="Skip credential validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=== BensBot Broker Validator ===")
    
    success = validate_all_brokers(
        config_path=args.config,
        credential_path=args.credentials,
        skip_credentials=args.skip_credentials
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
