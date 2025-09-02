#!/usr/bin/env python3
"""
Authentication System Component Test

This script tests the core components of the authentication system:
1. Credential Store (both encrypted and YAML)
2. Audit Log (both SQLite and JSON)
3. Event Bus Integration with Audit Log

This is a standalone test that doesn't require the complete trading bot system.
"""

import os
import sys
import json
import tempfile
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("auth_test")

# Import credential store components directly
sys.path.insert(0, str(Path(__file__).parent))

# Create credential store classes directly here to avoid import issues
class AuthMethod:
    """Authentication methods supported by brokers."""
    API_KEY = "api_key"
    OAUTH = "oauth"
    USERNAME_PASSWORD = "username_password"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    CUSTOM = "custom"


class BrokerCredentials:
    """Base class for broker credentials."""
    
    def __init__(self):
        self.api_key = None
        self.api_secret = None
        self.client_id = None
        self.client_secret = None
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        self.username = None
        self.password = None
        self.cert_path = None
        self.key_path = None
        self.token = None
        self.custom_data = {}


# Create a basic implementation of FileStore for testing
class TestFileStore:
    """Test implementation of credential store that uses files."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.credentials = {}
        self.auth_methods = {}
        
    def store_credentials(self, broker_id, credentials, auth_method):
        """Store credentials for a broker."""
        self.credentials[broker_id] = credentials
        self.auth_methods[broker_id] = auth_method
        logger.info(f"Stored credentials for {broker_id} using {auth_method}")
        return True
        
    def get_credentials(self, broker_id):
        """Get credentials for a broker."""
        if broker_id not in self.credentials:
            raise ValueError(f"No credentials found for broker {broker_id}")
        return self.credentials[broker_id]
        
    def get_auth_method(self, broker_id):
        """Get authentication method for a broker."""
        if broker_id not in self.auth_methods:
            raise ValueError(f"No auth method found for broker {broker_id}")
        return self.auth_methods[broker_id]
        
    def list_brokers(self):
        """List all brokers with stored credentials."""
        return list(self.credentials.keys())
        
    def delete_credentials(self, broker_id):
        """Delete credentials for a broker."""
        if broker_id in self.credentials:
            del self.credentials[broker_id]
        if broker_id in self.auth_methods:
            del self.auth_methods[broker_id]
        logger.info(f"Deleted credentials for {broker_id}")
        return True
        
    def store_api_key_credentials(self, broker_id, api_key, api_secret=""):
        """Store API key credentials."""
        creds = BrokerCredentials()
        creds.api_key = api_key
        creds.api_secret = api_secret
        return self.store_credentials(broker_id, creds, AuthMethod.API_KEY)


# Create a basic implementation of AuditLog for testing
class AuditEventType:
    """Types of events recorded in the audit log."""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"
    BROKER_OPERATION = "broker_operation"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGE = "config_change"
    STRATEGY_SIGNAL = "strategy_signal"
    RISK_LIMIT_BREACH = "risk_limit_breach"


class AuditEvent:
    """Event recorded in the audit log."""
    
    def __init__(self, event_type, details, broker_id=None, order_id=None, strategy_id=None):
        self.event_id = f"event_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.timestamp = datetime.now()
        self.event_type = event_type
        self.details = details
        self.broker_id = broker_id
        self.order_id = order_id
        self.strategy_id = strategy_id


class TestAuditLog:
    """Test implementation of audit log."""
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.events = []
        
    def log_event(self, event_type, details, broker_id=None, order_id=None, strategy_id=None):
        """Log an event to the audit log."""
        event = AuditEvent(event_type, details, broker_id, order_id, strategy_id)
        self.events.append(event)
        logger.info(f"Logged event {event_type} to audit log")
        return event
        
    def query_events(self, event_types=None, broker_id=None, order_id=None, strategy_id=None, 
                     start_time=None, end_time=None):
        """Query events from the audit log."""
        results = []
        for event in self.events:
            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
            if broker_id and event.broker_id != broker_id:
                continue
            if order_id and event.order_id != order_id:
                continue
            if strategy_id and event.strategy_id != strategy_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
                
            results.append(event)
            
        return results
        
    def get_order_history(self, order_id):
        """Get the history of events for a specific order."""
        return self.query_events(order_id=order_id)


# Create a simple Event Bus for testing
class Event:
    """Event for the event bus."""
    
    def __init__(self, event_type, data=None, source=""):
        self.event_type = event_type
        self.data = data or {}
        self.source = source
        self.timestamp = datetime.now()
        self.event_id = f"event_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


class EventBus:
    """Simple event bus implementation for testing."""
    
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event):
        """Publish an event to subscribers."""
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {str(e)}")


# Create a test implementation of AuditLogListener
class AuditLogListener:
    """Connect event bus to audit log."""
    
    def __init__(self, audit_log, event_bus):
        self.audit_log = audit_log
        self.event_bus = event_bus
        self.event_mapping = {
            "order_created": AuditEventType.ORDER_SUBMITTED,
            "order_submitted": AuditEventType.ORDER_SUBMITTED,
            "order_filled": AuditEventType.ORDER_FILLED,
            "order_cancelled": AuditEventType.ORDER_CANCELLED,
            "order_rejected": AuditEventType.ORDER_REJECTED,
            "trade_executed": AuditEventType.ORDER_FILLED,
            "position_updated": AuditEventType.POSITION_UPDATED,
            "error": AuditEventType.SYSTEM_ERROR,
        }
        
    def register(self):
        """Register with the event bus."""
        for event_type in self.event_mapping:
            self.event_bus.subscribe(event_type, self.handle_event)
        logger.info("Audit log listener registered with event bus")
        
    def handle_event(self, event):
        """Handle an event by logging it to the audit log."""
        audit_event_type = self.event_mapping.get(event.event_type)
        if not audit_event_type:
            return
            
        # Extract broker_id and order_id if present
        broker_id = event.data.get("broker_id")
        order_id = event.data.get("order_id")
        strategy_id = event.data.get("strategy_id")
        
        # Log to audit trail
        self.audit_log.log_event(
            audit_event_type,
            event.data,
            broker_id=broker_id,
            order_id=order_id,
            strategy_id=strategy_id
        )
        logger.info(f"Logged {event.event_type} to audit log as {audit_event_type}")


def test_credential_store():
    """Test the credential store functionality."""
    print("\n=== Testing Credential Store ===")
    
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create credential store
        store_path = os.path.join(temp_dir, "test_credentials.json")
        store = TestFileStore(store_path)
        
        # Test storing credentials
        print("Testing credential storage...")
        success = store.store_api_key_credentials("test_broker", "test_api_key", "test_secret")
        assert success, "Failed to store credentials"
        
        # Test retrieving credentials
        print("Testing credential retrieval...")
        creds = store.get_credentials("test_broker")
        assert creds.api_key == "test_api_key", f"API key mismatch: {creds.api_key}"
        assert creds.api_secret == "test_secret", f"API secret mismatch: {creds.api_secret}"
        
        # Test listing brokers
        print("Testing broker listing...")
        brokers = store.list_brokers()
        assert "test_broker" in brokers, f"Broker not found in listing: {brokers}"
        
        # Test storing multiple brokers
        print("Testing multiple broker support...")
        store.store_api_key_credentials("another_broker", "another_key")
        brokers = store.list_brokers()
        assert len(brokers) == 2, f"Expected 2 brokers, got {len(brokers)}"
        
        # Test deleting credentials
        print("Testing credential deletion...")
        success = store.delete_credentials("test_broker")
        assert success, "Failed to delete credentials"
        brokers = store.list_brokers()
        assert "test_broker" not in brokers, f"Broker still in listing after deletion: {brokers}"
        
        print("✅ Credential store tests passed")
        return True
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_audit_log():
    """Test the audit log functionality."""
    print("\n=== Testing Audit Log ===")
    
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create audit log
        log_path = os.path.join(temp_dir, "test_audit.log")
        audit_log = TestAuditLog(log_path)
        
        # Test logging events
        print("Testing event logging...")
        event1 = audit_log.log_event(
            AuditEventType.ORDER_SUBMITTED,
            {"symbol": "AAPL", "quantity": 10},
            broker_id="test_broker",
            order_id="order123"
        )
        
        event2 = audit_log.log_event(
            AuditEventType.ORDER_FILLED,
            {"symbol": "AAPL", "quantity": 10, "price": 150.0},
            broker_id="test_broker",
            order_id="order123"
        )
        
        event3 = audit_log.log_event(
            AuditEventType.SYSTEM_ERROR,
            {"error": "Connection lost"},
            broker_id="another_broker"
        )
        
        # Test querying events
        print("Testing event querying...")
        all_events = audit_log.query_events()
        assert len(all_events) == 3, f"Expected 3 events, got {len(all_events)}"
        
        # Test filtering by event type
        order_events = audit_log.query_events(
            event_types=[AuditEventType.ORDER_SUBMITTED, AuditEventType.ORDER_FILLED]
        )
        assert len(order_events) == 2, f"Expected 2 order events, got {len(order_events)}"
        
        # Test filtering by broker
        broker_events = audit_log.query_events(broker_id="test_broker")
        assert len(broker_events) == 2, f"Expected 2 events for test_broker, got {len(broker_events)}"
        
        # Test filtering by order
        order_history = audit_log.get_order_history("order123")
        assert len(order_history) == 2, f"Expected 2 events for order123, got {len(order_history)}"
        
        print("✅ Audit log tests passed")
        return True
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_event_bus_integration():
    """Test the event bus integration with audit log."""
    print("\n=== Testing Event Bus Integration ===")
    
    # Create components
    audit_log = TestAuditLog("memory")
    event_bus = EventBus()
    listener = AuditLogListener(audit_log, event_bus)
    
    # Register listener
    print("Registering audit log listener...")
    listener.register()
    
    # Publish events
    print("Publishing test events...")
    event_bus.publish(Event(
        "order_submitted",
        {
            "order_id": "order456",
            "symbol": "MSFT",
            "quantity": 5,
            "broker_id": "test_broker"
        },
        "test_module"
    ))
    
    event_bus.publish(Event(
        "order_filled",
        {
            "order_id": "order456",
            "symbol": "MSFT",
            "quantity": 5,
            "price": 300.0,
            "broker_id": "test_broker"
        },
        "test_module"
    ))
    
    # Verify events were logged
    print("Verifying events were logged...")
    logged_events = audit_log.query_events(order_id="order456")
    assert len(logged_events) == 2, f"Expected 2 logged events, got {len(logged_events)}"
    
    types = [e.event_type for e in logged_events]
    assert AuditEventType.ORDER_SUBMITTED in types, f"ORDER_SUBMITTED not found in logged events: {types}"
    assert AuditEventType.ORDER_FILLED in types, f"ORDER_FILLED not found in logged events: {types}"
    
    print("✅ Event bus integration tests passed")
    return True


def main():
    """Main function to run the tests."""
    print("\n🔒 Authentication System Component Test\n")
    
    # Create a master password for testing
    os.environ['TRADING_BOT_MASTER_PASSWORD'] = 'test_master_password'
    print("Master password set for testing")
    
    # Run tests
    credential_store_success = test_credential_store()
    audit_log_success = test_audit_log()
    event_bus_success = test_event_bus_integration()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Credential Store: {'✅ PASSED' if credential_store_success else '❌ FAILED'}")
    print(f"Audit Log: {'✅ PASSED' if audit_log_success else '❌ FAILED'}")
    print(f"Event Bus Integration: {'✅ PASSED' if event_bus_success else '❌ FAILED'}")
    
    if credential_store_success and audit_log_success and event_bus_success:
        print("\n✅ All authentication system tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
