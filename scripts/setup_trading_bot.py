#!/usr/bin/env python3
"""
Trading Bot Setup Script

This script initializes the trading bot with a secure authentication system
including credential storage and audit logging. It demonstrates how to:

1. Load broker configuration
2. Initialize secure credential storage
3. Set up audit logging with event bus integration
4. Create and configure the multi-broker manager
5. Connect to brokers

Usage:
    python setup_trading_bot.py --config config/broker_config.json
"""

import os
import argparse
import logging
from typing import Dict, Any

from trading_bot.brokers.auth_manager import (
    create_credential_store, 
    create_audit_log, 
    initialize_broker_credentials,
    create_audit_log_listener,
    initialize_auth_system,
    load_config
)
from trading_bot.brokers.broker_factory import create_broker_manager, load_from_config_file
from trading_bot.core.event_bus import get_global_event_bus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables for sensitive information"""
    # Get master password from environment or prompt user
    master_password = os.environ.get('TRADING_BOT_MASTER_PASSWORD')
    if not master_password:
        # In a real application, use a secure way to handle passwords
        # This is just for demonstration
        import getpass
        master_password = getpass.getpass("Enter master password for credential store: ")
        os.environ['TRADING_BOT_MASTER_PASSWORD'] = master_password


def initialize_trading_bot(config_path: str) -> Dict[str, Any]:
    """
    Initialize the trading bot with secure authentication and audit logging
    
    Args:
        config_path: Path to broker configuration file
        
    Returns:
        Dictionary with initialized components
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize the global event bus
    event_bus = get_global_event_bus()
    logger.info("Global event bus initialized")
    
    # Initialize complete auth system (credential store, audit log, and audit log listener)
    logger.info("Initializing authentication and audit logging system")
    credential_store, audit_log, audit_listener = initialize_auth_system(config)
    
    if not credential_store:
        logger.error("Failed to create credential store")
        return {}
    
    if not audit_log:
        logger.warning("Audit logging is not available")
    else:
        logger.info("Audit log initialized successfully")
        
    if not audit_listener:
        logger.warning("Audit log listener could not be registered with event bus")
    else:
        logger.info("Audit log listener registered with event bus - all trading events will be logged")
    
    # Initialize broker manager with secure components
    logger.info("Creating broker manager with secure components")
    broker_manager = create_broker_manager(config)
    
    # Connect to brokers
    logger.info("Connecting to brokers")
    connection_results = broker_manager.connect_all()
    for broker_id, success in connection_results.items():
        if success:
            logger.info(f"Successfully connected to {broker_id}")
        else:
            logger.warning(f"Failed to connect to {broker_id}")
    
    # Get available brokers
    available_brokers = broker_manager.get_available_brokers()
    logger.info(f"Available brokers: {available_brokers}")
    
    return {
        "broker_manager": broker_manager,
        "credential_store": credential_store,
        "audit_log": audit_log,
        "audit_listener": audit_listener,
        "event_bus": event_bus,
        "config": config
    }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Initialize Trading Bot with Secure Authentication')
    parser.add_argument('--config', type=str, default='config/broker_config.json',
                      help='Path to broker configuration file')
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Initialize trading bot
    components = initialize_trading_bot(args.config)
    
    if not components:
        logger.error("Failed to initialize trading bot")
        return
    
    broker_manager = components.get("broker_manager")
    if not broker_manager:
        logger.error("No broker manager available")
        return
    
    # Display initialization information
    logger.info("Trading Bot initialized successfully")
    logger.info(f"Primary broker: {broker_manager.primary_broker_id}")
    logger.info(f"Active broker: {broker_manager.active_broker_id}")
    
    # Display audit log status
    audit_log = components.get("audit_log")
    audit_listener = components.get("audit_listener")
    if audit_log and audit_listener:
        logger.info("Audit logging system is active - all trading operations will be logged")
    elif audit_log:
        logger.info("Audit log is available but not connected to event bus - manual logging only")
    else:
        logger.warning("Audit logging is not available")
    
    # Display asset routing
    logger.info("Asset Routing:")
    for asset_type, broker_id in broker_manager.asset_routing.items():
        logger.info(f"  {asset_type.name}: {broker_id}")
    
    # Example: Generate sample events to demonstrate audit logging
    event_bus = components.get("event_bus")
    if event_bus and audit_listener:
        logger.info("Generating sample events to demonstrate audit logging")
        from trading_bot.core.constants import EventType
        
        # Simulate system startup event
        event_bus.create_and_publish(
            EventType.SYSTEM_STARTED,
            {"component": "trading_bot", "version": "1.0.0"},
            "setup_script"
        )
        
        # Simulate strategy started event
        event_bus.create_and_publish(
            EventType.STRATEGY_STARTED,
            {"strategy_id": "forex_momentum", "parameters": {"timeframe": "1h"}},
            "setup_script"
        )
        
        logger.info("Sample events published to event bus and logged to audit log")
    
    logger.info("Setup complete - trading bot is ready")



if __name__ == "__main__":
    main()
