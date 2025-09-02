"""
Tests for the TradeAuditLog module
"""

import os
import unittest
import tempfile
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from trading_bot.brokers.trade_audit_log import (
    TradeAuditLog, JsonFileAuditLog, SqliteAuditLog, 
    AuditEventType, AuditLogFactory
)


class TestTradeAuditLog(unittest.TestCase):
    """Test cases for trade audit log implementations"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.json_log_dir = os.path.join(self.test_dir.name, "json_logs")
        self.sqlite_db = os.path.join(self.test_dir.name, "audit.db")
        
        # Ensure directories exist
        os.makedirs(self.json_log_dir, exist_ok=True)
        
        # Sample test event details
        self.test_order_details = {
            "symbol": "AAPL",
            "quantity": 10,
            "order_type": "market",
            "order_side": "buy",
            "price": 150.0,
            "time_in_force": "day"
        }

    def tearDown(self):
        """Clean up after tests"""
        self.test_dir.cleanup()

    def test_json_file_audit_log(self):
        """Test JsonFileAuditLog functionality"""
        # Create log
        log = JsonFileAuditLog(self.json_log_dir)
        
        # Log an event
        event_id = log.log_event(
            AuditEventType.ORDER_SUBMITTED,
            self.test_order_details,
            broker_id="etrade",
            order_id="order123"
        )
        
        # Verify event was logged
        self.assertIsNotNone(event_id)
        
        # Get events
        events = log.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], AuditEventType.ORDER_SUBMITTED.value)
        self.assertEqual(events[0]["broker_id"], "etrade")
        self.assertEqual(events[0]["order_id"], "order123")
        self.assertEqual(events[0]["details"]["symbol"], "AAPL")
        
        # Get specific event
        event = log.get_event(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_id"], event_id)
        
        # Log more events for the same order
        log.log_event(
            AuditEventType.ORDER_FILLED,
            {"filled_quantity": 10, "fill_price": 149.95},
            broker_id="etrade",
            order_id="order123"
        )
        
        # Test order history
        order_history = log.get_order_history("order123")
        self.assertEqual(len(order_history), 2)
        self.assertEqual(order_history[0]["event_type"], AuditEventType.ORDER_SUBMITTED.value)
        self.assertEqual(order_history[1]["event_type"], AuditEventType.ORDER_FILLED.value)
        
        # Test filtering
        filtered_events = log.get_events({"broker_id": "etrade"})
        self.assertEqual(len(filtered_events), 2)
        
        # Verify file was created
        log_files = os.listdir(self.json_log_dir)
        self.assertEqual(len(log_files), 1)
        
        # Check file contents
        log_file_path = os.path.join(self.json_log_dir, log_files[0])
        with open(log_file_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)

    def test_sqlite_audit_log(self):
        """Test SqliteAuditLog functionality"""
        # Create log
        log = SqliteAuditLog(self.sqlite_db)
        
        # Log an event
        event_id = log.log_event(
            AuditEventType.ORDER_SUBMITTED,
            self.test_order_details,
            broker_id="alpaca",
            order_id="order456"
        )
        
        # Verify event was logged
        self.assertIsNotNone(event_id)
        
        # Get events
        events = log.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], AuditEventType.ORDER_SUBMITTED.value)
        self.assertEqual(events[0]["broker_id"], "alpaca")
        self.assertEqual(events[0]["details"]["symbol"], "AAPL")
        
        # Log multiple event types
        log.log_event(
            AuditEventType.BROKER_CONNECTED,
            {"status": "success"},
            broker_id="alpaca"
        )
        
        log.log_event(
            AuditEventType.POSITION_OPENED,
            {"symbol": "AAPL", "quantity": 10, "price": 150.0},
            broker_id="alpaca",
            order_id="order456"
        )
        
        # Test filtering by event type
        filtered_by_type = log.get_events({"event_type": AuditEventType.BROKER_CONNECTED.value})
        self.assertEqual(len(filtered_by_type), 1)
        self.assertEqual(filtered_by_type[0]["event_type"], AuditEventType.BROKER_CONNECTED.value)
        
        # Test order history
        order_history = log.get_order_history("order456")
        self.assertEqual(len(order_history), 2)
        
        # Verify database file was created
        self.assertTrue(os.path.exists(self.sqlite_db))
        
        # Test direct database access
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 3)
        conn.close()

    def test_audit_log_factory(self):
        """Test AuditLogFactory methods"""
        # Test JSON file log creation
        json_log = AuditLogFactory.create_json_file_log(self.json_log_dir)
        self.assertIsInstance(json_log, JsonFileAuditLog)
        
        # Test SQLite log creation
        sqlite_log = AuditLogFactory.create_sqlite_log(self.sqlite_db)
        self.assertIsInstance(sqlite_log, SqliteAuditLog)


if __name__ == "__main__":
    unittest.main()
