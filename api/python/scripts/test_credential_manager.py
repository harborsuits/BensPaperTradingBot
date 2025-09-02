#!/usr/bin/env python3
"""
Credential Manager Test Script

Demonstrates the usage of the credential manager with various safety features,
showing clear separation between configuration and credentials.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import credential manager
from core.credential_manager import (
    CredentialManager, CredentialError,
    is_dev_environment, is_live_trading_enabled, safe_to_trade_live
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("credential_test")

def demo_credential_manager(args):
    """Demonstrate credential manager functionality"""
    
    print("\n=== Credential Manager Demo ===\n")
    
    # Print environment information
    env_type = "DEVELOPMENT" if is_dev_environment() else "PRODUCTION"
    live_trading = is_live_trading_enabled()
    safe_live = safe_to_trade_live()
    
    print(f"Environment Type: {env_type}")
    print(f"Live Trading Enabled: {live_trading}")
    print(f"Safe to Trade Live: {safe_live}")
    
    # Initialize credential manager
    cred_manager = CredentialManager(
        credential_dir=args.cred_dir,
        use_mock_in_dev=args.use_mock,
        dev_mode=is_dev_environment()
    )
    
    # 1. Demonstrate configuration vs credentials separation
    print("\n--- Configuration vs Credentials Separation ---")
    print("Configuration (broker_config.json):")
    print("  - Defines broker endpoints, limits, priorities")
    print("  - References credential files by broker ID")
    print("  - Safe to commit to version control")
    print("\nCredentials (credential_manager.py):")
    print("  - Loads secrets from files or environment")
    print("  - Never stored in code or config")
    print("  - Multiple fallback mechanisms")
    
    # 2. Test with different broker types
    print("\n--- Broker Credential Testing ---")
    
    # Test paper broker (no credentials needed)
    print("\nPaper Broker:")
    paper_creds = cred_manager.get_broker_credentials("paper_simulator", "paper")
    print(f"  Credentials: {paper_creds if paper_creds else 'None needed'}")
    print(f"  Source: {cred_manager.get_credential_source('paper_simulator')}")
    print(f"  Live Credentials: {cred_manager.has_live_credentials('paper_simulator', 'paper')}")
    
    # Test Tradier broker
    print("\nTradier Broker:")
    try:
        tradier_creds = cred_manager.get_broker_credentials(
            "tradier", "tradier", 
            required_fields=["api_key", "account_id"],
            allow_missing=args.allow_missing
        )
        source = cred_manager.get_credential_source("tradier")
        print(f"  Source: {source}")
        if tradier_creds:
            # Show safely masked credentials
            masked = {k: f"{v[:4]}{'*' * (len(v)-4)}" if k in ["api_key", "api_secret"] and len(v) > 4 else v 
                     for k, v in tradier_creds.items()}
            print(f"  Credentials: {masked}")
        else:
            print("  No credentials found")
        print(f"  Live Credentials: {cred_manager.has_live_credentials('tradier', 'tradier')}")
    except CredentialError as e:
        print(f"  Error: {str(e)}")
    
    # 3. Safety Integration Demo
    if args.show_safety:
        print("\n--- Integration with Safeguard System ---")
        
        # Demonstrate how credential manager supports robustness
        print("\nRobustness Features:")
        print("  1. Clear distinction between paper and live trading")
        print("  2. Multiple credential fallbacks prevent runtime failures")
        print("  3. Extra safeguards in development environment")
        print("  4. Safe credential transition with hot reload")
        print("  5. Environment-aware trading permissions")
        
        # Show position manager safeguard integration
        print("\nPosition Manager Safeguard Integration:")
        print("  - Position verification only runs with valid credentials")
        print("  - Broker connections monitored for authorization issues")
        print("  - State backups created before credential changes")
        print("  - Reconciliation handles multi-broker scenarios")
    
    # 4. Working with Templates
    if args.create_templates:
        print("\n--- Credential Templates ---")
        
        for broker_type in ["tradier", "alpaca", "interactive_brokers"]:
            template_path = cred_manager.create_template_file(broker_type)
            if template_path:
                with open(template_path, 'r') as f:
                    template = json.load(f)
                print(f"\n{broker_type.title()} Template:")
                print(json.dumps(template, indent=2))
    
    # 5. Demonstrate environment variable overrides
    print("\n--- Environment Variable Overrides ---")
    print("Examples:")
    print("  export BENBOT_BROKERS_TRADIER_CREDENTIALS_API_KEY=your_key")
    print("  export BENBOT_BROKERS_TRADIER_CREDENTIALS_ACCOUNT_ID=your_account")
    
    # Check for any currently set override variables
    env_overrides = []
    for key in os.environ:
        if key.startswith("BENBOT_BROKERS_") and "CREDENTIALS" in key:
            # Mask the value for security
            value = os.environ[key]
            if "KEY" in key or "SECRET" in key or "PASSWORD" in key:
                value = f"{value[:4]}{'*' * (len(value)-4)}" if len(value) > 4 else "****"
            env_overrides.append((key, value))
    
    if env_overrides:
        print("\nActive Credential Environment Variables:")
        for var, value in env_overrides:
            print(f"  {var}={value}")
    else:
        print("\nNo credential environment variables currently set")
    
    return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BensBot Credential Manager Demo")
    parser.add_argument("--cred-dir", default="config/credentials", help="Credential directory")
    parser.add_argument("--use-mock", action="store_true", help="Use mock credentials in dev mode")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing credentials")
    parser.add_argument("--show-safety", action="store_true", help="Show safety integration details")
    parser.add_argument("--create-templates", action="store_true", help="Create credential templates")
    args = parser.parse_args()
    
    return demo_credential_manager(args)

if __name__ == "__main__":
    sys.exit(main())
