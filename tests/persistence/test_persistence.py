#!/usr/bin/env python3
"""
Tests for persistence layer

This module provides tests for the repository and state recovery components.
"""

import unittest
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
import pymongo
import redis
import json

from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.persistence.mongo_repository import MongoRepository
from trading_bot.persistence.redis_repository import RedisRepository
from trading_bot.persistence.order_repository import OrderRepository, OrderModel
from trading_bot.persistence.fill_repository import FillRepository, FillModel
from trading_bot.persistence.position_repository import PositionRepository, PositionModel
from trading_bot.persistence.pnl_repository import PnLRepository, PnLModel
from trading_bot.persistence.idempotency import IdempotencyManager
from trading_bot.persistence.recovery_manager import RecoveryManager
from trading_bot.persistence.event_handlers import PersistenceEventHandler
from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import (
    OrderAcknowledged, OrderFilled, OrderPartialFill, OrderRejected, OrderCancelled,
    PositionUpdate, PortfolioEquityUpdate, EventType
)


class TestPersistenceLayer(unittest.TestCase):
    """Test suite for persistence layer"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Mock connection manager
        cls.conn_manager = MagicMock(spec=ConnectionManager)
        cls.mongo_client = MagicMock(spec=pymongo.MongoClient)
        cls.redis_client = MagicMock(spec=redis.Redis)
        cls.conn_manager.get_mongo_client.return_value = cls.mongo_client
        cls.conn_manager.get_redis_client.return_value = cls.redis_client
        
        # Create database and collection mocks
        cls.mongo_db = MagicMock()
        cls.mongo_client.__getitem__.return_value = cls.mongo_db
        
        # Set up collections for each repository
        cls.orders_collection = MagicMock()
        cls.fills_collection = MagicMock()
        cls.positions_collection = MagicMock()
        cls.pnl_collection = MagicMock()
        cls.idempotency_collection = MagicMock()
        
        cls.mongo_db.__getitem__.side_effect = lambda x: {
            'orders': cls.orders_collection,
            'fills': cls.fills_collection,
            'positions': cls.positions_collection,
            'pnl_records': cls.pnl_collection,
            'idempotency': cls.idempotency_collection
        }.get(x, MagicMock())
        
        # Mock EventBus
        cls.event_bus = MagicMock(spec=EventBus)
    
    def test_order_repository(self):
        """Test order repository operations"""
        order_repo = OrderRepository(self.conn_manager)
        
        # Mock collection operations
        self.orders_collection.insert_one.return_value.inserted_id = "mock_id"
        self.orders_collection.find_one.return_value = {
            "_id": "mock_id",
            "internal_id": "test_order_1",
            "broker": "tradier",
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day",
            "status": "new",
            "broker_order_id": "broker_order_1",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "filled_quantity": 0
        }
        self.orders_collection.find.return_value = [self.orders_collection.find_one.return_value]
        
        # Create and save an order
        order = OrderModel(
            internal_id="test_order_1",
            broker="tradier",
            symbol="AAPL",
            quantity=10,
            side="buy",
            order_type="market",
            time_in_force="day",
            status="new",
            broker_order_id="broker_order_1"
        )
        
        # Test save operation
        order_id = order_repo.save_order(order)
        self.assertEqual(order_id, "test_order_1")
        self.orders_collection.insert_one.assert_called_once()
        
        # Test find by internal ID
        retrieved_order = order_repo.find_by_internal_id("test_order_1")
        self.assertIsNotNone(retrieved_order)
        self.assertEqual(retrieved_order.internal_id, "test_order_1")
        
        # Test update status
        order_repo.update_status("test_order_1", "filled")
        self.orders_collection.update_one.assert_called()
        
        # Test fetch open orders
        open_orders = order_repo.fetch_open_orders()
        self.assertEqual(len(open_orders), 1)
    
    def test_fill_repository(self):
        """Test fill repository operations"""
        fill_repo = FillRepository(self.conn_manager)
        
        # Mock collection operations
        self.fills_collection.insert_one.return_value.inserted_id = "mock_id"
        self.fills_collection.find.return_value = [{
            "_id": "mock_id",
            "order_internal_id": "test_order_1",
            "fill_qty": 10,
            "fill_price": 150.0,
            "timestamp": datetime.now().isoformat(),
            "event_type": "fill",
            "fill_id": "test_fill_1",
            "symbol": "AAPL",
            "broker": "tradier"
        }]
        
        # Create an order fill event
        fill_event = OrderFilled(
            order_id="test_order_1",
            symbol="AAPL",
            broker="tradier",
            side="buy",
            total_qty=10,
            avg_fill_price=150.0
        )
        
        # Test record fill
        fill_id = fill_repo.record_fill(fill_event)
        self.assertIsNotNone(fill_id)
        self.fills_collection.insert_one.assert_called_once()
        
        # Test find by order ID
        fills = fill_repo.find_by_order_id("test_order_1")
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].order_internal_id, "test_order_1")
        
        # Test find by symbol
        fills = fill_repo.find_by_symbol("AAPL")
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].symbol, "AAPL")
    
    def test_position_repository(self):
        """Test position repository operations"""
        position_repo = PositionRepository(self.conn_manager)
        
        # Mock collection and Redis operations
        self.positions_collection.insert_one.return_value.inserted_id = "mock_id"
        self.positions_collection.find_one.return_value = {
            "_id": "mock_id",
            "position_id": "tradier:AAPL",
            "symbol": "AAPL",
            "quantity": 10,
            "avg_cost": 150.0,
            "broker": "tradier",
            "last_updated": datetime.now().isoformat(),
            "unrealized_pnl": 50.0,
            "realized_pnl": 0.0
        }
        self.positions_collection.find.return_value = [self.positions_collection.find_one.return_value]
        
        # Mock Redis methods
        self.redis_client.set.return_value = True
        self.redis_client.get.return_value = json.dumps({
            "position_id": "tradier:AAPL",
            "symbol": "AAPL",
            "quantity": 10,
            "avg_cost": 150.0,
            "broker": "tradier",
            "last_updated": datetime.now().isoformat(),
            "unrealized_pnl": 50.0,
            "realized_pnl": 0.0
        })
        
        # Create a position
        position = PositionModel(
            symbol="AAPL",
            quantity=10,
            avg_cost=150.0,
            broker="tradier",
            unrealized_pnl=50.0
        )
        
        # Test save position
        position_id = position_repo.save_position(position)
        self.assertEqual(position_id, "tradier:AAPL")
        self.positions_collection.insert_one.assert_called_once()
        
        # Test find by position ID
        retrieved_position = position_repo.find_by_position_id("tradier:AAPL")
        self.assertIsNotNone(retrieved_position)
        self.assertEqual(retrieved_position.symbol, "AAPL")
        
        # Test update from fill
        position_repo.update_from_fill("AAPL", "tradier", 155.0, 5)
        self.positions_collection.update_one.assert_called()
        
        # Test find non-zero positions
        positions = position_repo.find_non_zero_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, "AAPL")
    
    def test_pnl_repository(self):
        """Test PnL repository operations"""
        pnl_repo = PnLRepository(self.conn_manager)
        
        # Mock collection and Redis operations
        self.pnl_collection.insert_one.return_value.inserted_id = "mock_id"
        self.pnl_collection.find_one.return_value = {
            "_id": "mock_id",
            "timestamp": datetime.now().isoformat(),
            "total_equity": 10000.0,
            "unrealized_pnl": 500.0,
            "realized_pnl": 200.0,
            "record_type": "snapshot"
        }
        self.pnl_collection.find.return_value = [self.pnl_collection.find_one.return_value]
        
        # Create a PnL snapshot
        pnl = PnLModel(
            timestamp=datetime.now(),
            total_equity=10000.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0
        )
        
        # Test record snapshot
        result = pnl_repo.record_snapshot(pnl)
        self.assertIsNotNone(result)
        self.pnl_collection.insert_one.assert_called_once()
        
        # Test get latest snapshot
        latest = pnl_repo.get_latest_snapshot()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.total_equity, 10000.0)
        
        # Test get snapshots in range
        start_time = datetime.now().replace(hour=0, minute=0, second=0)
        end_time = datetime.now()
        snapshots = pnl_repo.get_snapshots_in_range(start_time, end_time)
        self.assertEqual(len(snapshots), 1)
    
    def test_idempotency_manager(self):
        """Test idempotency manager operations"""
        idempotency_manager = IdempotencyManager(self.conn_manager)
        
        # Mock collection operations
        self.idempotency_collection.insert_one.return_value.inserted_id = "mock_id"
        self.idempotency_collection.find_one.return_value = {
            "_id": "mock_id",
            "idempotency_key": "test_key_1",
            "operation_type": "place_order",
            "broker": "tradier",
            "params": {"symbol": "AAPL", "quantity": 10, "side": "buy"},
            "created_at": datetime.now().isoformat(),
            "result": None
        }
        self.idempotency_collection.find.return_value = [self.idempotency_collection.find_one.return_value]
        
        # Test register operation
        key = idempotency_manager.register_operation(
            "place_order", "tradier", {"symbol": "AAPL", "quantity": 10, "side": "buy"}
        )
        self.assertIsNotNone(key)
        self.idempotency_collection.insert_one.assert_called_once()
        
        # Test get operation
        operation = idempotency_manager.get_operation("test_key_1")
        self.assertIsNotNone(operation)
        self.assertEqual(operation["operation_type"], "place_order")
        
        # Test record result
        idempotency_manager.record_result("test_key_1", {"order_id": "order_123"})
        self.idempotency_collection.update_one.assert_called()
        
        # Test find pending operations
        pending_ops = idempotency_manager.find_pending_operations()
        self.assertEqual(len(pending_ops), 1)
    
    @patch('trading_bot.persistence.recovery_manager.RecoveryManager.recover_open_orders')
    @patch('trading_bot.persistence.recovery_manager.RecoveryManager.recover_positions')
    @patch('trading_bot.persistence.recovery_manager.RecoveryManager.recover_latest_pnl')
    def test_recovery_manager(self, mock_recover_pnl, mock_recover_positions, mock_recover_orders):
        """Test recovery manager operations"""
        # Set up mocks
        mock_recover_orders.return_value = [MagicMock(internal_id="test_order_1")]
        mock_recover_positions.return_value = [MagicMock(broker="tradier", symbol="AAPL")]
        mock_recover_pnl.return_value = {"total_equity": 10000.0}
        
        # Create recovery manager
        recovery_manager = RecoveryManager(
            self.conn_manager,
            self.event_bus
        )
        
        # Test full state recovery
        result = recovery_manager.recover_full_state()
        self.assertIsNotNone(result)
        self.assertEqual(result["open_orders_recovered"], 1)
        self.assertEqual(result["positions_recovered"], 1)
        self.assertTrue(result["pnl_recovered"])
        
        # Verify calls to component recovery methods
        mock_recover_orders.assert_called_once()
        mock_recover_positions.assert_called_once()
        mock_recover_pnl.assert_called_once()
    
    def test_persistence_event_handler(self):
        """Test persistence event handler operations"""
        # Create persistence event handler
        handler = PersistenceEventHandler(self.conn_manager, self.event_bus)
        
        # Test event subscription
        self.event_bus.subscribe.assert_called()
        
        # Test handling order acknowledged event
        order_ack_event = OrderAcknowledged(
            order_id="test_order_1",
            broker="tradier",
            symbol="AAPL",
            quantity=10,
            side="buy",
            order_type="market",
            broker_order_id="broker_order_1"
        )
        handler._handle_order_acknowledged(order_ack_event)
        
        # Test handling order filled event
        order_filled_event = OrderFilled(
            order_id="test_order_1",
            broker="tradier",
            symbol="AAPL",
            side="buy",
            total_qty=10,
            avg_fill_price=150.0
        )
        handler._handle_order_filled(order_filled_event)
        
        # Test handling position update event
        position_update_event = PositionUpdate(
            symbol="AAPL",
            broker="tradier",
            quantity=10,
            avg_cost=150.0
        )
        handler._handle_position_update(position_update_event)
        
        # Test handling portfolio equity update event
        portfolio_update_event = PortfolioEquityUpdate(
            total_equity=10000.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0
        )
        handler._handle_portfolio_equity_update(portfolio_update_event)


if __name__ == '__main__':
    unittest.main()
