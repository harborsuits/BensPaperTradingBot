#!/usr/bin/env python3
"""
Credential Rotation Utility

Provides zero-downtime credential rotation for broker API keys and secrets.
This tool enables secure credential updates without disrupting trading operations.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import credential manager (with fallback if not available)
try:
    from core.credential_manager import CredentialManager, CredentialError
    CREDENTIAL_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIAL_MANAGER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("credential_rotation")

class RotationError(Exception):
    """Exception raised for credential rotation errors"""
    pass

def backup_credentials(credential_path: str, broker_id: str) -> str:
    """
    Create a backup of the existing credentials
    
    Args:
        credential_path: Path to credentials directory
        broker_id: Broker identifier
        
    Returns:
        Path to backup file
    """
    cred_file = os.path.join(credential_path, f"{broker_id}.json")
    if not os.path.exists(cred_file):
        logger.warning(f"No existing credential file found for {broker_id}")
        return ""
    
    # Create backup directory if it doesn't exist
    backup_dir = os.path.join(credential_path, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"{broker_id}_{timestamp}.json")
    
    try:
        # Read current credentials
        with open(cred_file, 'r') as f:
            current_creds = json.load(f)
        
        # Write to backup file
        with open(backup_file, 'w') as f:
            json.dump(current_creds, f, indent=2)
        
        # Set secure permissions
        os.chmod(backup_file, 0o600)
        
        logger.info(f"Created credential backup at {backup_file}")
        return backup_file
        
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        return ""

def validate_new_credentials(
    credential_path: str,
    broker_id: str,
    broker_type: str,
    new_credentials: Dict[str, Any]
) -> List[str]:
    """
    Validate new credentials without applying them
    
    Args:
        credential_path: Path to credentials directory
        broker_id: Broker identifier
        broker_type: Type of broker
        new_credentials: New credentials to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Define required fields by broker type
    required_fields = []
    if broker_type == "alpaca":
        required_fields = ["api_key", "api_secret"]
    elif broker_type == "tradier":
        required_fields = ["api_key", "account_id"]
    elif broker_type == "interactive_brokers":
        required_fields = ["tws_host", "tws_port"]
    elif broker_type == "paper":
        # Paper broker doesn't need credentials
        return []
    else:
        errors.append(f"Unknown broker type: {broker_type}")
        return errors
    
    # Check required fields
    for field in required_fields:
        if field not in new_credentials:
            errors.append(f"Missing required field: {field}")
    
    return errors

def test_new_credentials(
    broker_id: str,
    broker_type: str,
    new_credentials: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Test new credentials without applying them
    
    Args:
        broker_id: Broker identifier
        broker_type: Type of broker
        new_credentials: New credentials to test
        
    Returns:
        Tuple of (success, message)
    """
    # Implement broker-specific connection tests
    if broker_type == "paper":
        return (True, "Paper broker doesn't require credentials")
    
    # Here we would implement actual connection tests for each broker type
    # For now, just simulate a test based on credential presence
    if broker_type == "alpaca":
        if "api_key" in new_credentials and "api_secret" in new_credentials:
            # In a real implementation, we would test actual API connection
            return (True, "Simulated Alpaca credential test passed")
        else:
            return (False, "Missing required Alpaca credentials")
            
    elif broker_type == "tradier":
        if "api_key" in new_credentials and "account_id" in new_credentials:
            # In a real implementation, we would test actual API connection
            return (True, "Simulated Tradier credential test passed")
        else:
            return (False, "Missing required Tradier credentials")
            
    elif broker_type == "interactive_brokers":
        if "tws_host" in new_credentials and "tws_port" in new_credentials:
            # In a real implementation, we would test actual API connection
            return (True, "Simulated Interactive Brokers credential test passed")
        else:
            return (False, "Missing required Interactive Brokers credentials")
    
    return (False, f"Unknown broker type: {broker_type}")

def rotate_credentials(
    credential_path: str,
    broker_id: str,
    broker_type: str,
    new_credentials: Dict[str, Any],
    skip_test: bool = False,
    force: bool = False
) -> bool:
    """
    Perform credential rotation with backup and verification
    
    Args:
        credential_path: Path to credentials directory
        broker_id: Broker identifier
        broker_type: Type of broker
        new_credentials: New credentials to apply
        skip_test: Skip connection test
        force: Force rotation even if tests fail
        
    Returns:
        True if rotation was successful
    """
    logger.info(f"Starting credential rotation for {broker_id} ({broker_type})")
    
    # Skip for paper broker
    if broker_type == "paper":
        logger.info("Paper broker doesn't require credentials, skipping rotation")
        return True
    
    # Create backup of existing credentials
    backup_file = backup_credentials(credential_path, broker_id)
    
    # Validate new credentials
    errors = validate_new_credentials(credential_path, broker_id, broker_type, new_credentials)
    if errors and not force:
        logger.error("Validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    # Test new credentials
    if not skip_test:
        success, message = test_new_credentials(broker_id, broker_type, new_credentials)
        if not success and not force:
            logger.error(f"Credential test failed: {message}")
            return False
        logger.info(f"Credential test result: {message}")
    
    # Write new credentials
    cred_file = os.path.join(credential_path, f"{broker_id}.json")
    try:
        with open(cred_file, 'w') as f:
            json.dump(new_credentials, f, indent=2)
        
        # Set secure permissions
        os.chmod(cred_file, 0o600)
        
        logger.info(f"Successfully rotated credentials for {broker_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write new credentials: {str(e)}")
        
        # Try to restore backup if available
        if backup_file and os.path.exists(backup_file):
            logger.info(f"Attempting to restore backup from {backup_file}")
            try:
                with open(backup_file, 'r') as f:
                    backup_creds = json.load(f)
                
                with open(cred_file, 'w') as f:
                    json.dump(backup_creds, f, indent=2)
                
                logger.info("Successfully restored backup")
            except Exception as restore_err:
                logger.error(f"Failed to restore backup: {str(restore_err)}")
        
        return False

def interactive_rotation(credential_path: str, broker_id: str, broker_type: str) -> bool:
    """
    Interactive credential rotation prompt
    
    Args:
        credential_path: Path to credentials directory
        broker_id: Broker identifier
        broker_type: Type of broker
        
    Returns:
        True if rotation was successful
    """
    print(f"\n=== Interactive Credential Rotation for {broker_id} ({broker_type}) ===\n")
    
    # Get template for broker type
    template = {}
    if broker_type == "alpaca":
        template = {
            "api_key": "",
            "api_secret": "",
            "base_url": "https://paper-api.alpaca.markets"
        }
    elif broker_type == "tradier":
        template = {
            "api_key": "",
            "account_id": ""
        }
    elif broker_type == "interactive_brokers":
        template = {
            "tws_host": "127.0.0.1",
            "tws_port": 7497,
            "client_id": 1
        }
    elif broker_type == "paper":
        print("Paper broker doesn't require credentials")
        return True
    else:
        print(f"Unknown broker type: {broker_type}")
        return False
    
    # Prompt for each credential field
    new_credentials = {}
    for field, default_value in template.items():
        if field in ["api_key", "api_secret", "password"]:
            prompt = f"Enter {field} (sensitive): "
        else:
            prompt = f"Enter {field} [{default_value}]: "
        
        value = input(prompt)
        
        # Use default if empty
        if not value and default_value:
            value = default_value
        
        # Store value
        new_credentials[field] = value
    
    # Confirm rotation
    print("\nNew credentials:")
    for field, value in new_credentials.items():
        if field in ["api_key", "api_secret", "password"]:
            # Mask sensitive fields
            masked = '*' * (len(value) - 4) + value[-4:] if len(value) > 4 else '****'
            print(f"  {field}: {masked}")
        else:
            print(f"  {field}: {value}")
    
    confirm = input("\nApply these credentials? (y/n): ")
    if confirm.lower() != 'y':
        print("Rotation cancelled")
        return False
    
    # Perform rotation
    return rotate_credentials(credential_path, broker_id, broker_type, new_credentials)

def load_broker_config(config_path: str) -> Dict[str, Any]:
    """Load broker configuration"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load broker config: {str(e)}")
        return {}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BensBot Credential Rotation Utility")
    parser.add_argument("--broker-config", default="config/broker_config.json", help="Path to broker configuration")
    parser.add_argument("--credential-dir", default="config/credentials", help="Path to credentials directory")
    parser.add_argument("--broker", required=True, help="Broker ID to rotate credentials for")
    parser.add_argument("--interactive", action="store_true", help="Interactive credential entry")
    parser.add_argument("--skip-test", action="store_true", help="Skip connection test")
    parser.add_argument("--force", action="store_true", help="Force rotation even if tests fail")
    
    # Credential arguments for non-interactive mode
    parser.add_argument("--api-key", help="API key (for Alpaca, Tradier)")
    parser.add_argument("--api-secret", help="API secret (for Alpaca)")
    parser.add_argument("--account-id", help="Account ID (for Tradier)")
    parser.add_argument("--host", help="Host (for Interactive Brokers)")
    parser.add_argument("--port", type=int, help="Port (for Interactive Brokers)")
    
    args = parser.parse_args()
    
    # Load broker configuration to get broker type
    broker_config = load_broker_config(args.broker_config)
    
    if not broker_config or "brokers" not in broker_config:
        logger.error("Invalid broker configuration")
        return 1
    
    if args.broker not in broker_config["brokers"]:
        logger.error(f"Broker {args.broker} not found in configuration")
        return 1
    
    broker_type = broker_config["brokers"][args.broker].get("broker_type")
    if not broker_type:
        logger.error(f"No broker type found for {args.broker}")
        return 1
    
    # Check if paper broker
    if broker_type == "paper":
        logger.info("Paper broker doesn't require credentials")
        return 0
    
    # Interactive mode
    if args.interactive:
        success = interactive_rotation(args.credential_dir, args.broker, broker_type)
        return 0 if success else 1
    
    # Non-interactive mode
    new_credentials = {}
    
    # Build credentials based on broker type
    if broker_type == "alpaca":
        if not args.api_key or not args.api_secret:
            logger.error("API key and API secret are required for Alpaca")
            return 1
        new_credentials = {
            "api_key": args.api_key,
            "api_secret": args.api_secret,
            "base_url": "https://paper-api.alpaca.markets"  # Default to paper
        }
    
    elif broker_type == "tradier":
        if not args.api_key or not args.account_id:
            logger.error("API key and account ID are required for Tradier")
            return 1
        new_credentials = {
            "api_key": args.api_key,
            "account_id": args.account_id
        }
    
    elif broker_type == "interactive_brokers":
        host = args.host or "127.0.0.1"
        port = args.port or 7497
        new_credentials = {
            "tws_host": host,
            "tws_port": port,
            "client_id": 1
        }
    
    # Perform rotation
    success = rotate_credentials(
        args.credential_dir, 
        args.broker, 
        broker_type, 
        new_credentials,
        skip_test=args.skip_test,
        force=args.force
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
