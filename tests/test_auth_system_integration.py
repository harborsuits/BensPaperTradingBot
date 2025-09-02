#!/usr/bin/env python3
"""
Authentication System Integration Tests

This module tests the complete authentication system, including:
- Credential Store
- Audit Log
- Event Bus Integration
- Broker Integration
"""

import os
import sys
import unittest
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trading_bot.brokers.credential_store import (
    AuthMethod, CredentialFactory, BrokerCredentials
)
from trading_bot.brokers.trade_audit_log import (
    AuditEventType, AuditEvent
)
from trading_bot.brokers.auth_manager import (
    initialize_auth_system, create_credential_store, 
    create_audit_log, create_audit_log_listener
)
from trading_bot.core.event_bus import (
    EventBus, Event
)
from trading_bot.core.constants import EventType
from trading_bot.core.audit_log_listener import AuditLogListener


class TestAuthSystemIntegration(unittest.TestCase):
    """Test the complete authentication system integration"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Set environment variable for master password
        os.environ['TRADING_BOT_MASTER_PASSWORD'] = 'test_master_password'
        
        # Create test configuration
        self.config = {
            'credential_store': {
                'type': 'encrypted',
                'path': os.path.join(self.temp_dir, 'credentials.enc')
            },
            'audit_log': {
                'enabled': True,
                'type': 'sqlite',
                'path': os.path.join(self.temp_dir, 'audit.db')
            },
            'brokers': {
                'test_broker': {
                    'enabled': True,
                    'api_key': 'test_api_key',
                    'api_secret': 'test_api_secret',
                    'sandbox': True
                }
            }
        }
        
        # Save config to temporary file
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        
        # Create event bus
        self.event_bus = EventBus()
    
    def tearDown(self):
        """Clean up after each test"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Remove environment variable
        if 'TRADING_BOT_MASTER_PASSWORD' in os.environ:
            del os.environ['TRADING_BOT_MASTER_PASSWORD']
    
    def test_initialize_auth_system(self):
        """Test initializing the complete auth system"""
        # Initialize auth system
        credential_store, audit_log, audit_listener = initialize_auth_system(self.config)
        
        # Verify components are created
        self.assertIsNotNone(credential_store)
        self.assertIsNotNone(audit_log)
        self.assertIsNotNone(audit_listener)
        
        # Verify credential store is properly initialized
        brokers = credential_store.list_brokers()
        self.assertEqual(len(brokers), 1)
        self.assertIn('test_broker', brokers)
        
        # Verify credentials can be retrieved
        auth_method = credential_store.get_auth_method('test_broker')
        self.assertEqual(auth_method, AuthMethod.API_KEY)
        
        credentials = credential_store.get_credentials('test_broker')
        self.assertEqual(credentials.api_key, 'test_api_key')
        self.assertEqual(credentials.api_secret, 'test_api_secret')
    
    def test_audit_log_integration(self):
        """Test audit log integration"""
        # Initialize auth system
        credential_store, audit_log, audit_listener = initialize_auth_system(self.config)
        
        # Create a test event
        event = Event(
            event_type=EventType.ORDER_SUBMITTED,
            data={
                'order_id': 'test_order_123',
                'symbol': 'AAPL',
                'quantity': 10,
                'price': 150.0,
                'broker_id': 'test_broker'
            },
            source='test_module'
        )
        
        # Publish event to event bus
        self.event_bus.publish(event)
        
        # Give the audit log listener time to process the event
        # In a real test, we would use a mock or wait for a callback
        
        # Query the audit log for the event
        events = audit_log.query_events(
            event_types=[AuditEventType.ORDER_SUBMITTED],
            order_id='test_order_123'
        )
        
        # Verify the event was logged
        self.assertEqual(len(events), 1)
        
        logged_event = events[0]
        self.assertEqual(logged_event.event_type, AuditEventType.ORDER_SUBMITTED)
        self.assertEqual(logged_event.order_id, 'test_order_123')
        self.assertEqual(logged_event.broker_id, 'test_broker')
        self.assertIn('symbol', logged_event.details)
        self.assertEqual(logged_event.details['symbol'], 'AAPL')
    
    def test_event_mapping(self):
        """Test event mapping between event bus and audit log"""
        # Create audit log listener with test audit log
        audit_log = MagicMock()
        listener = AuditLogListener(audit_log, self.event_bus)
        listener.register()
        
        # Create different event types and verify mapping
        event_mappings = [
            # Order events
            (EventType.ORDER_CREATED, AuditEventType.ORDER_SUBMITTED),
            (EventType.ORDER_SUBMITTED, AuditEventType.ORDER_SUBMITTED),
            (EventType.ORDER_FILLED, AuditEventType.ORDER_FILLED),
            (EventType.ORDER_CANCELLED, AuditEventType.ORDER_CANCELLED),
            (EventType.ORDER_REJECTED, AuditEventType.ORDER_REJECTED),
            
            # Trade events
            (EventType.TRADE_EXECUTED, AuditEventType.ORDER_FILLED),
            (EventType.TRADE_CLOSED, AuditEventType.POSITION_CLOSED),
            
            # Strategy events
            (EventType.STRATEGY_STARTED, AuditEventType.SYSTEM_ERROR),
            (EventType.SIGNAL_GENERATED, AuditEventType.STRATEGY_SIGNAL),
        ]
        
        for event_type, audit_type in event_mappings:
            # Reset mock
            audit_log.reset_mock()
            
            # Create event
            event = Event(
                event_type=event_type,
                data={'test_key': 'test_value'},
                source='test_module'
            )
            
            # Publish event
            self.event_bus.publish(event)
            
            # Verify audit log was called with correct event type
            audit_log.log_event.assert_called_once()
            call_args = audit_log.log_event.call_args[0]
            self.assertEqual(call_args[0], audit_type)
    
    def test_credential_store_broker_integration(self):
        """Test credential store integration with broker"""
        # Initialize credential store
        credential_store = create_credential_store(self.config)
        
        # Mock broker class
        class MockBroker:
            def __init__(self):
                self.authenticated = False
                self.credentials = None
            
            def authenticate(self, credentials):
                self.authenticated = True
                self.credentials = credentials
                return True
        
        # Create mock broker
        broker = MockBroker()
        
        # Get credentials from store and authenticate broker
        credentials = credential_store.get_credentials('test_broker')
        result = broker.authenticate(credentials)
        
        # Verify authentication
        self.assertTrue(result)
        self.assertTrue(broker.authenticated)
        self.assertEqual(broker.credentials.api_key, 'test_api_key')
        self.assertEqual(broker.credentials.api_secret, 'test_api_secret')
    
    def test_different_storage_types(self):
        """Test different storage types for credential store and audit log"""
        # Create YAML credential store config
        yaml_config = self.config.copy()
        yaml_config['credential_store'] = {
            'type': 'yaml',
            'path': os.path.join(self.temp_dir, 'credentials.yml')
        }
        
        # Create JSON audit log config
        json_config = self.config.copy()
        json_config['audit_log'] = {
            'enabled': True,
            'type': 'json',
            'directory': os.path.join(self.temp_dir, 'audit_logs')
        }
        
        # Test YAML credential store
        yaml_store = create_credential_store(yaml_config)
        self.assertIsNotNone(yaml_store)
        
        # Add test credentials
        yaml_store.store_api_key_credentials(
            'yaml_broker', 'yaml_api_key', 'yaml_api_secret'
        )
        
        # Test JSON audit log
        json_log = create_audit_log(json_config)
        self.assertIsNotNone(json_log)
        
        # Log test event
        json_log.log_event(
            AuditEventType.ORDER_SUBMITTED,
            {'symbol': 'AAPL', 'quantity': 10},
            broker_id='test_broker',
            order_id='test_order_456'
        )
        
        # Verify credentials were stored
        credentials = yaml_store.get_credentials('yaml_broker')
        self.assertEqual(credentials.api_key, 'yaml_api_key')
        self.assertEqual(credentials.api_secret, 'yaml_api_secret')
        
        # Verify event was logged
        events = json_log.query_events(
            event_types=[AuditEventType.ORDER_SUBMITTED],
            order_id='test_order_456'
        )
        self.assertEqual(len(events), 1)


if __name__ == '__main__':
    unittest.main()
