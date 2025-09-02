#!/usr/bin/env python3
"""
Secure Broker Testing Script

This script demonstrates how to use the secure authentication system
with different brokers, one at a time. It:

1. Sets up the authentication system with encrypted credentials
2. Connects to brokers using secure credential retrieval
3. Logs all operations to the audit log automatically

Usage:
    python test_secure_brokers.py --broker tradier
    python test_secure_brokers.py --broker alpaca
    python test_secure_brokers.py --broker all
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("secure_broker_test")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from trading_bot.brokers.auth_manager import (
    initialize_auth_system, load_config as load_auth_config
)
from trading_bot.brokers.broker_factory import create_broker_manager
from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.brokers.trade_audit_log import AuditEventType


def check_environment():
    """Check if environment is set up correctly"""
    if 'TRADING_BOT_MASTER_PASSWORD' not in os.environ:
        logger.error("TRADING_BOT_MASTER_PASSWORD environment variable is not set")
        logger.error("Please set it before running this script:")
        logger.error("export TRADING_BOT_MASTER_PASSWORD='your_secure_password'")
        return False
    return True


def initialize_broker_system(broker_type):
    """
    Initialize the broker system for the specified broker type
    
    Args:
        broker_type: 'tradier', 'alpaca', or 'all'
        
    Returns:
        Tuple of (broker_manager, credential_store, audit_log, audit_listener)
    """
    # Map broker type to config file
    config_paths = {
        'tradier': 'config/tradier_config.json',
        'alpaca': 'config/alpaca_config.json',
        'all': 'config/multi_broker_config.json'
    }
    
    config_path = config_paths.get(broker_type)
    if not config_path:
        logger.error(f"Unknown broker type: {broker_type}")
        logger.error(f"Must be one of: {', '.join(config_paths.keys())}")
        return None, None, None, None
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None, None, None, None
    
    # Load broker configuration
    logger.info(f"Loading broker configuration from {config_path}")
    try:
        broker_config = load_auth_config(config_path)
    except Exception as e:
        logger.error(f"Error loading broker configuration: {str(e)}")
        return None, None, None, None
    
    # Initialize the global event bus
    event_bus = get_global_event_bus()
    logger.info("Global event bus initialized")
    
    # Initialize the complete authentication system
    logger.info("Initializing authentication and audit system")
    credential_store, audit_log, audit_listener = initialize_auth_system(broker_config)
    
    if not credential_store:
        logger.error("Failed to initialize credential store")
        return None, None, None, None
    
    # Initialize broker manager with credentials
    logger.info("Creating broker manager with secure credential store")
    broker_manager = create_broker_manager(broker_config)
    
    return broker_manager, credential_store, audit_log, audit_listener


def test_broker_connections(broker_manager):
    """
    Test connections to all brokers in the manager
    
    Args:
        broker_manager: The broker manager instance
        
    Returns:
        List of available broker IDs
    """
    logger.info("Testing broker connections...")
    connection_results = broker_manager.connect_all()
    
    available_brokers = []
    for broker_id, success in connection_results.items():
        if success:
            logger.info(f"✅ Successfully connected to {broker_id}")
            available_brokers.append(broker_id)
        else:
            logger.warning(f"❌ Failed to connect to {broker_id}")
    
    return available_brokers


def check_account_info(broker_manager, broker_id):
    """Get account information for a specific broker"""
    try:
        broker = broker_manager.brokers.get(broker_id)
        if not broker or not broker.is_connected():
            logger.warning(f"Broker {broker_id} is not connected")
            return None
        
        logger.info(f"Getting account information for {broker_id}...")
        account_info = broker.get_account_info()
        logger.info(f"Account information for {broker_id}:")
        
        # Format account information
        for key, value in account_info.items():
            logger.info(f"  {key}: {value}")
        
        return account_info
    except Exception as e:
        logger.error(f"Error getting account information for {broker_id}: {str(e)}")
        return None


def get_broker_positions(broker_manager, broker_id):
    """Get positions for a specific broker"""
    try:
        broker = broker_manager.brokers.get(broker_id)
        if not broker or not broker.is_connected():
            logger.warning(f"Broker {broker_id} is not connected")
            return []
        
        logger.info(f"Getting positions for {broker_id}...")
        positions = broker.get_positions()
        logger.info(f"Positions for {broker_id}: {len(positions)}")
        
        # Format positions
        for position in positions:
            logger.info(f"  {position.symbol}: {position.quantity} @ {position.avg_price}")
        
        return positions
    except Exception as e:
        logger.error(f"Error getting positions for {broker_id}: {str(e)}")
        return []


def simulate_order_events(broker_manager, event_bus, broker_id):
    """
    Simulate order events to demonstrate audit logging
    
    Args:
        broker_manager: The broker manager instance
        event_bus: The event bus instance
        broker_id: The broker ID to use for events
    """
    logger.info(f"Simulating order events for {broker_id}...")
    
    # Create a test order
    order_id = f"test_order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    symbol = "AAPL"
    quantity = 10
    price = 150.0
    
    # Publish order created event
    logger.info(f"Creating order {order_id} for {symbol}")
    event_bus.publish(Event(
        event_type=EventType.ORDER_CREATED,
        data={
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "broker_id": broker_id,
            "status": "created"
        },
        source="test_script"
    ))
    
    # Simulate a small delay
    time.sleep(1)
    
    # Publish order submitted event
    logger.info(f"Submitting order {order_id} to {broker_id}")
    event_bus.publish(Event(
        event_type=EventType.ORDER_SUBMITTED,
        data={
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "broker_id": broker_id,
            "status": "submitted"
        },
        source="test_script"
    ))
    
    # Simulate a small delay
    time.sleep(1)
    
    # Publish order filled event
    logger.info(f"Order {order_id} filled")
    event_bus.publish(Event(
        event_type=EventType.ORDER_FILLED,
        data={
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "fill_price": 149.95,
            "broker_id": broker_id,
            "status": "filled"
        },
        source="test_script"
    ))
    
    return order_id


def check_audit_log(audit_log, order_id=None, broker_id=None):
    """
    Check the audit log for events
    
    Args:
        audit_log: The audit log instance
        order_id: Optional order ID to filter events
        broker_id: Optional broker ID to filter events
    """
    logger.info("Checking audit log for events...")
    
    # Build query parameters
    query_params = {}
    if order_id:
        query_params["order_id"] = order_id
    if broker_id:
        query_params["broker_id"] = broker_id
    
    # Query recent events
    try:
        events = audit_log.query_events(**query_params)
        logger.info(f"Found {len(events)} events in the audit log")
        
        # Display recent events
        for i, event in enumerate(events[-10:] if len(events) > 10 else events):
            logger.info(f"Event {i+1}: {event.event_type.value} at {event.timestamp}")
            logger.info(f"  Broker: {event.broker_id}")
            logger.info(f"  Order: {event.order_id}")
            
            # Display event details for the first few events
            if i < 3:
                details_str = json.dumps(event.details, indent=2)
                if len(details_str) > 500:
                    details_str = details_str[:500] + "..."
                logger.info(f"  Details: {details_str}")
        
        # If order_id provided, get the full order history
        if order_id:
            order_history = audit_log.get_order_history(order_id)
            logger.info(f"Order history for {order_id}: {len(order_history)} events")
            
            for i, event in enumerate(order_history):
                logger.info(f"  {i+1}. {event.event_type.value} at {event.timestamp}")
        
        return events
    except Exception as e:
        logger.error(f"Error querying audit log: {str(e)}")
        return []


def test_broker(broker_type):
    """
    Test a specific broker with the secure authentication system
    
    Args:
        broker_type: 'tradier', 'alpaca', or 'all'
    """
    logger.info(f"Testing {broker_type} broker with secure authentication...")
    
    # Initialize broker system
    broker_manager, credential_store, audit_log, audit_listener = initialize_broker_system(broker_type)
    if not broker_manager:
        logger.error("Failed to initialize broker system. Exiting.")
        return
    
    # Get the event bus
    event_bus = get_global_event_bus()
    
    # Test broker connections
    available_brokers = test_broker_connections(broker_manager)
    if not available_brokers:
        logger.error("No brokers available. Please check your credentials.")
        return
    
    # Display broker system information
    logger.info("\n=== Broker System Status ===")
    logger.info(f"Primary broker: {broker_manager.primary_broker_id}")
    logger.info(f"Active broker: {broker_manager.active_broker_id}")
    logger.info(f"Asset routing:")
    for asset_type, broker_id in broker_manager.asset_routing.items():
        logger.info(f"  {asset_type}: {broker_id}")
    
    # Test each available broker
    for broker_id in available_brokers:
        logger.info(f"\n=== Testing {broker_id} Broker ===")
        
        # Check account information
        check_account_info(broker_manager, broker_id)
        
        # Get positions
        get_broker_positions(broker_manager, broker_id)
        
        # Simulate order events
        order_id = simulate_order_events(broker_manager, event_bus, broker_id)
        
        # Check audit log for the broker
        logger.info(f"\n=== Audit Log for {broker_id} ===")
        check_audit_log(audit_log, order_id=order_id, broker_id=broker_id)
    
    # Check all audit events
    logger.info("\n=== All Audit Events ===")
    check_audit_log(audit_log)
    
    # Disconnect from brokers
    logger.info("\n=== Disconnecting from Brokers ===")
    for broker_id in available_brokers:
        try:
            broker = broker_manager.brokers.get(broker_id)
            if broker and broker.is_connected():
                broker.disconnect()
                logger.info(f"Disconnected from {broker_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from {broker_id}: {str(e)}")
    
    logger.info("\n=== Test Completed Successfully ===")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test secure broker authentication')
    parser.add_argument('--broker', type=str, choices=['tradier', 'alpaca', 'all'],
                        default='all', help='Broker to test')
    args = parser.parse_args()
    
    # Print welcome message
    print("\n🔒 Secure Broker Authentication Test")
    print("======================================\n")
    
    # Check environment
    if not check_environment():
        return 1
    
    # Test specified broker
    try:
        test_broker(args.broker)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
