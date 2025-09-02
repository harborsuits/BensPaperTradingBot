"""
Integration tests for the new credential store and audit log with MultiBrokerManager
"""

import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
from datetime import datetime

from trading_bot.brokers.broker_interface import (
    BrokerInterface, BrokerCredentials, Order, Position, AssetType,
    OrderType, OrderSide, OrderStatus, TimeInForce
)
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.brokers.credential_store import (
    CredentialStore, YamlFileStore, CredentialFactory, AuthMethod
)
from trading_bot.brokers.trade_audit_log import (
    TradeAuditLog, SqliteAuditLog, AuditEventType
)


class MockBroker(BrokerInterface):
    """Mock broker implementation for testing"""
    
    def __init__(self, broker_name="MockBroker"):
        self.connected = False
        self.broker_name = broker_name
        self.orders = {}
        self.positions = []
        
    def is_connected(self):
        return self.connected
        
    def connect(self, credentials):
        self.connected = True
        return True
        
    def disconnect(self):
        self.connected = False
        return True
        
    def get_broker_name(self):
        return self.broker_name
        
    def place_order(self, order):
        order_copy = Order(
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.side,
            order_type=order.order_type,
            time_in_force=order.time_in_force,
            asset_type=order.asset_type
        )
        order_copy.order_id = "order_" + str(len(self.orders) + 1)
        order_copy.status = OrderStatus.FILLED
        order_copy.filled_quantity = order.quantity
        order_copy.fill_price = 150.0
        
        self.orders[order_copy.order_id] = order_copy
        return order_copy
        
    def cancel_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
        
    def get_positions(self):
        return self.positions
        
    def get_account_info(self):
        return [{"account_id": "mock_account", "equity": 100000.0}]
        
    def get_supported_asset_types(self):
        return [AssetType.STOCK, AssetType.OPTION]
    
    # Implement remaining required methods with minimal functionality
    def is_market_open(self):
        return True
        
    def get_next_market_open(self):
        return datetime.now()
        
    def get_next_market_close(self):
        return datetime.now()
        
    def get_account_balance(self):
        return {"cash": 50000.0, "equity": 100000.0}
        
    def get_quote(self, symbol, asset_type=AssetType.STOCK):
        return {"symbol": symbol, "bid": 149.5, "ask": 150.0, "last": 149.75}
        
    def get_order_status(self, order_id):
        return self.orders.get(order_id)
        
    def get_orders(self):
        return list(self.orders.values())
        
    def get_bars(self, symbol, timeframe, start, end=None, limit=None, asset_type=AssetType.STOCK):
        return []
    
    @property
    def supports_extended_hours(self):
        return False
        
    @property
    def supports_fractional_shares(self):
        return False


class TestBrokerIntegration(unittest.TestCase):
    """Integration tests for broker components"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Create credential store
        self.cred_file = os.path.join(self.test_dir.name, "credentials.yml")
        self.credential_store = YamlFileStore(self.cred_file)
        
        # Create audit log
        self.db_file = os.path.join(self.test_dir.name, "audit.db")
        self.audit_log = SqliteAuditLog(self.db_file)
        
        # Create broker manager with credential store and audit log
        self.manager = MultiBrokerManager(
            credential_store=self.credential_store,
            audit_log=self.audit_log
        )
        
        # Create mock brokers
        self.etrade_broker = MockBroker("E*TRADE")
        self.alpaca_broker = MockBroker("Alpaca")
        
        # Create credentials
        self.etrade_credentials = CredentialFactory.create_api_key_credentials(
            "etrade_key", "etrade_secret", {"account_id": "etrade_account"}
        )
        
        self.alpaca_credentials = CredentialFactory.create_api_key_credentials(
            "alpaca_key", "alpaca_secret", {"account_id": "alpaca_account"}
        )

    def tearDown(self):
        """Clean up after tests"""
        self.test_dir.cleanup()

    def test_integration(self):
        """Test integration of all components"""
        # Add brokers to manager
        self.assertTrue(
            self.manager.add_broker(
                "etrade", 
                self.etrade_broker, 
                self.etrade_credentials,
                make_primary=True
            )
        )
        
        self.assertTrue(
            self.manager.add_broker(
                "alpaca", 
                self.alpaca_broker, 
                self.alpaca_credentials
            )
        )
        
        # Verify brokers were added
        brokers = self.manager.get_available_brokers()
        self.assertEqual(len(brokers), 2)
        self.assertIn("etrade", brokers)
        self.assertIn("alpaca", brokers)
        
        # Verify credentials were stored
        stored_brokers = self.credential_store.list_brokers()
        self.assertEqual(len(stored_brokers), 2)
        self.assertIn("etrade", stored_brokers)
        self.assertIn("alpaca", stored_brokers)
        
        # Connect brokers
        connect_results = self.manager.connect_all()
        self.assertTrue(connect_results["etrade"])
        self.assertTrue(connect_results["alpaca"])
        
        # Check audit log for connection events
        connection_events = self.audit_log.get_events(
            {"event_type": AuditEventType.BROKER_CONNECTED.value}
        )
        self.assertEqual(len(connection_events), 2)
        
        # Set asset routing
        self.manager.set_asset_routing(AssetType.STOCK, "etrade")
        self.manager.set_asset_routing(AssetType.OPTION, "alpaca")
        
        # Place an order
        order = Order(
            symbol="AAPL",
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            asset_type=AssetType.STOCK
        )
        
        result_order = self.manager.place_order(order)
        
        # Verify order was placed
        self.assertEqual(result_order.status, OrderStatus.FILLED)
        self.assertEqual(result_order.filled_quantity, 10)
        
        # Check audit log for order events
        order_events = self.audit_log.get_events(
            {"event_type": AuditEventType.ORDER_SUBMITTED.value}
        )
        self.assertGreaterEqual(len(order_events), 1)
        
        # Place another order with a different asset type (should route to alpaca)
        option_order = Order(
            symbol="AAPL",
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            asset_type=AssetType.OPTION
        )
        
        option_result = self.manager.place_order(option_order)
        
        # Get order history
        order_history = self.manager.get_order_history(result_order.order_id)
        self.assertGreaterEqual(len(order_history), 1)
        
        # Get broker-specific events
        etrade_events = self.manager.get_broker_events("etrade")
        self.assertGreaterEqual(len(etrade_events), 1)
        
        # Disconnect all brokers
        disconnect_results = self.manager.disconnect_all()
        self.assertTrue(disconnect_results["etrade"])
        self.assertTrue(disconnect_results["alpaca"])


if __name__ == "__main__":
    unittest.main()
