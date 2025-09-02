#!/usr/bin/env python3
"""
Broker Connection Validator

Validates broker connections and credentials using the configuration system.
Provides detailed diagnostics and validation to ensure brokers are correctly configured.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core configuration
from trading_bot.core.simple_config import (
    load_config, get_nested_value, ConfigError,
    ConfigFileNotFoundError, ConfigParseError, ConfigValidationError
)

# Import broker components
from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.brokers.paper_broker import PaperBroker
from trading_bot.brokers.tradier_broker import TradierBroker
from trading_bot.brokers.alpaca_broker import AlpacaBroker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("broker_validator")


class BrokerValidationError(Exception):
    """Exception raised when broker validation fails"""
    pass


def load_broker_config(config_path: str) -> Dict[str, Any]:
    """
    Load broker configuration from file
    
    Args:
        config_path: Path to broker configuration file
        
    Returns:
        Broker configuration dictionary
    """
    try:
        config = load_config(config_path)
        logger.info(f"Loaded broker configuration from {config_path}")
        return config
    except ConfigError as e:
        logger.error(f"Failed to load broker configuration: {str(e)}")
        raise BrokerValidationError(f"Failed to load broker configuration: {str(e)}")


def load_broker_credentials(
    broker_id: str, 
    credential_path: str,
    broker_type: str
) -> Dict[str, Any]:
    """
    Load broker credentials from file
    
    Args:
        broker_id: Broker identifier
        credential_path: Path to credentials directory
        broker_type: Type of broker
        
    Returns:
        Credentials dictionary
    """
    # Paper broker doesn't need credentials
    if broker_type == "paper":
        return {}
    
    # Construct path to credentials file
    cred_file = os.path.join(credential_path, f"{broker_id}.json")
    
    # Check if credentials file exists
    if not os.path.exists(cred_file):
        logger.warning(f"Credentials file not found: {cred_file}")
        return {}
    
    # Load credentials
    try:
        with open(cred_file, 'r') as f:
            credentials = json.load(f)
        logger.info(f"Loaded credentials for broker: {broker_id}")
        return credentials
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in credentials file {cred_file}: {str(e)}")
        raise BrokerValidationError(f"Invalid JSON in credentials file {cred_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load credentials for broker {broker_id}: {str(e)}")
        raise BrokerValidationError(f"Failed to load credentials for broker {broker_id}: {str(e)}")


def validate_broker_config(broker_config: Dict[str, Any]) -> List[str]:
    """
    Validate broker configuration structure
    
    Args:
        broker_config: Broker configuration dictionary
        
    Returns:
        List of validation errors
    """
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
        
        # Check asset classes
        if "asset_classes" in broker_conf:
            asset_classes = broker_conf["asset_classes"]
            if not isinstance(asset_classes, list):
                errors.append(f"Broker {broker_id} has invalid asset_classes (must be a list)")
            else:
                for asset_class in asset_classes:
                    if asset_class not in ["equity", "option", "future", "forex", "crypto"]:
                        errors.append(f"Broker {broker_id} has invalid asset class: {asset_class}")
    
    # Check asset routing if present
    if "asset_routing" in broker_config:
        asset_routing = broker_config["asset_routing"]
        if not isinstance(asset_routing, dict):
            errors.append("Invalid asset_routing (must be a dictionary)")
        else:
            for asset_class, broker_id in asset_routing.items():
                if asset_class not in ["equity", "option", "future", "forex", "crypto"]:
                    errors.append(f"Invalid asset class in routing: {asset_class}")
                if broker_id not in broker_config["brokers"]:
                    errors.append(f"Asset {asset_class} routed to unknown broker: {broker_id}")
    
    return errors


def check_environment_overrides() -> List[Tuple[str, str]]:
    """
    Check for broker-related environment variable overrides
    
    Returns:
        List of (variable, value) pairs
    """
    overrides = []
    for key, value in os.environ.items():
        if key.startswith("BENBOT_BROKERS_") or key.startswith("BENBOT_DEFAULT_BROKER"):
            overrides.append((key, value))
    
    return overrides


def validate_broker_credentials(
    broker_id: str,
    broker_config: Dict[str, Any],
    credentials: Dict[str, Any]
) -> List[str]:
    """
    Validate broker credentials
    
    Args:
        broker_id: Broker identifier
        broker_config: Broker configuration
        credentials: Broker credentials
        
    Returns:
        List of validation errors
    """
    errors = []
    broker_type = broker_config.get("broker_type")
    
    # Skip if broker is disabled
    if not broker_config.get("enabled", False):
        return []
    
    # Paper broker doesn't need credentials
    if broker_type == "paper":
        return []
    
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
    
    return errors


def test_broker_connection(
    broker_id: str,
    broker_config: Dict[str, Any],
    credentials: Dict[str, Any],
    quick_check: bool = False
) -> Tuple[bool, str]:
    """
    Test connection to a broker
    
    Args:
        broker_id: Broker identifier
        broker_config: Broker configuration
        credentials: Broker credentials
        quick_check: If True, only do a basic connection test
        
    Returns:
        Tuple of (success, message)
    """
    broker_type = broker_config.get("broker_type")
    
    # Skip if broker is disabled
    if not broker_config.get("enabled", False):
        return (True, "Broker is disabled, skipping connection test")
    
    # Paper broker always connects
    if broker_type == "paper":
        return (True, "Paper broker simulated connection successful")
    
    # Here you would instantiate the actual broker and test connection
    # Since this is just a validation script, we'll simulate success/failure
    # based on the presence of credentials
    
    # For now, just simulate connections based on credentials
    if broker_type == "alpaca":
        if credentials.get("api_key") and credentials.get("api_secret"):
            return (True, "Simulated Alpaca connection successful")
        else:
            return (False, "Missing Alpaca credentials")
    
    elif broker_type == "tradier":
        if credentials.get("api_key") and credentials.get("account_id"):
            return (True, "Simulated Tradier connection successful")
        else:
            return (False, "Missing Tradier credentials")
    
    elif broker_type == "interactive_brokers":
        if credentials.get("tws_host") and credentials.get("tws_port"):
            return (True, "Simulated Interactive Brokers connection successful")
        else:
            return (False, "Missing Interactive Brokers credentials")
    
    return (False, f"Unknown broker type: {broker_type}")


def validate_all_brokers(
    config_path: str, 
    test_connections: bool = False,
    quick_check: bool = False
) -> bool:
    """
    Validate all brokers in configuration
    
    Args:
        config_path: Path to broker configuration file
        test_connections: If True, test connections to brokers
        quick_check: If True, only do basic validation
        
    Returns:
        True if validation successful, False otherwise
    """
    try:
        # Load broker configuration
        broker_config = load_broker_config(config_path)
        
        # Validate broker configuration structure
        config_errors = validate_broker_config(broker_config)
        if config_errors:
            print("\n❌ Broker configuration has errors:")
            for error in config_errors:
                print(f"  - {error}")
            return False
        
        print("\n✅ Broker configuration structure is valid")
        
        # Check environment overrides
        env_overrides = check_environment_overrides()
        if env_overrides:
            print("\nEnvironment variable overrides found:")
            for var, value in env_overrides:
                print(f"  {var}={value}")
        
        # Get credentials path
        credential_path = get_nested_value(broker_config, "credential_path", "config/credentials")
        
        # Create credentials directory if it doesn't exist
        os.makedirs(credential_path, exist_ok=True)
        
        # Validate each broker
        brokers = broker_config.get("brokers", {})
        enabled_brokers = [b for b, c in brokers.items() if c.get("enabled", False)]
        disabled_brokers = [b for b, c in brokers.items() if not c.get("enabled", False)]
        
        print(f"\nFound {len(enabled_brokers)} enabled broker(s) and {len(disabled_brokers)} disabled broker(s)")
        
        all_valid = True
        all_connection_errors = []
        
        # Process each broker
        for broker_id, broker_conf in brokers.items():
            broker_type = broker_conf.get("broker_type", "unknown")
            enabled = broker_conf.get("enabled", False)
            
            # Skip disabled brokers in quick check mode
            if quick_check and not enabled:
                continue
            
            print(f"\n{'✓' if enabled else '✗'} Broker: {broker_id} ({broker_type})")
            
            # Load credentials
            credentials = load_broker_credentials(broker_id, credential_path, broker_type)
            
            # Validate credentials
            cred_errors = validate_broker_credentials(broker_id, broker_conf, credentials)
            if cred_errors:
                print("  ❌ Credential validation failed:")
                for error in cred_errors:
                    print(f"    - {error}")
                all_valid = False
            else:
                print("  ✅ Credentials validation passed")
            
            # Test connection if requested and broker is enabled
            if test_connections and enabled:
                print("  🔄 Testing connection...")
                success, message = test_broker_connection(broker_id, broker_conf, credentials, quick_check)
                if success:
                    print(f"  ✅ Connection test passed: {message}")
                else:
                    print(f"  ❌ Connection test failed: {message}")
                    all_connection_errors.append(f"{broker_id}: {message}")
                    all_valid = False
        
        # Summary
        print("\n=== Broker Validation Summary ===")
        if all_valid:
            print("✅ All broker validations passed")
        else:
            print("❌ Some broker validations failed")
            
            # Show connection errors
            if all_connection_errors:
                print("\nConnection Errors:")
                for error in all_connection_errors:
                    print(f"  - {error}")
        
        return all_valid
        
    except BrokerValidationError as e:
        print(f"\n❌ Broker validation error: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BensBot Broker Validator")
    parser.add_argument("--config", default="config/broker_config.json", help="Path to broker configuration file")
    parser.add_argument("--test-connections", action="store_true", help="Test connections to enabled brokers")
    parser.add_argument("--quick", action="store_true", help="Quick check of enabled brokers only")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=== BensBot Broker Validator ===")
    
    success = validate_all_brokers(
        config_path=args.config,
        test_connections=args.test_connections,
        quick_check=args.quick
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
