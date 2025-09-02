"""
Integration tests for persistence and recovery functionality.

Tests the interaction between broker components, persistence layer,
and recovery mechanisms to ensure proper crash resilience.
"""

import os
import pytest
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest import mock

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import EventType
from trading_bot.brokers.paper.adapter import PaperTradeAdapter, PaperTradeConfig
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.persistence.connection_manager import ConnectionManager
from trading_bot.persistence.order_repository import OrderRepository
from trading_bot.persistence.position_repository import PositionRepository
from trading_bot.persistence.idempotency import IdempotencyManager
from trading_bot.persistence.recovery_manager import RecoveryManager


class TestPersistenceRecovery:
    """Integration tests for persistence and recovery functionality."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a real event bus for testing."""
        return EventBus()
    
    @pytest.fixture
    def connection_manager(self):
        """Create a connection manager for testing.
        
        Note: This uses real MongoDB and Redis if available, otherwise mocks.
        """
        mongodb_uri = os.environ.get('TEST_MONGODB_URI', 'mongodb://localhost:27017')
        redis_uri = os.environ.get('TEST_REDIS_URI', 'redis://localhost:6379/1')
        db_name = 'bensbot_test'
        
        try:
            # Try to create a real connection manager
            connection_manager = ConnectionManager(
                mongodb_uri=mongodb_uri,
                redis_uri=redis_uri,
                database_name=db_name
            )
            
            # Test MongoDB connection
            try:
                connection_manager.get_mongo_client().server_info()
            except Exception as e:
                # MongoDB not available, use mock
                print(f"MongoDB not available: {e}")
                connection_manager.get_mongo_client = mock.MagicMock()
                connection_manager.get_mongo_database = mock.MagicMock()
                connection_manager._mongo_client = mock.MagicMock()
                connection_manager._mongo_db = mock.MagicMock()
            
            # Test Redis connection
            try:
                connection_manager.get_redis_client().ping()
            except Exception as e:
                # Redis not available, use mock
                print(f"Redis not available: {e}")
                connection_manager.get_redis_client = mock.MagicMock()
                connection_manager._redis_client = mock.MagicMock()
            
            return connection_manager
            
        except Exception as e:
            # If connection setup fails, use mock
            print(f"Error creating connection manager: {e}")
            return mock.MagicMock(spec=ConnectionManager)
    
    @pytest.fixture
    def repositories(self, connection_manager):
        """Create repositories for testing."""
        order_repo = OrderRepository(connection_manager)
        position_repo = PositionRepository(connection_manager)
        idempotency_manager = IdempotencyManager(connection_manager)
        
        # Delete all existing data to start fresh
        try:
            mongo_db = connection_manager.get_mongo_database()
            if mongo_db:
                mongo_db.orders.delete_many({})
                mongo_db.positions.delete_many({})
                mongo_db.idempotency.delete_many({})
        except:
            # If using mocks, this might fail
            pass
        
        try:
            redis = connection_manager.get_redis_client()
            if redis:
                # Clear test database (db 1)
                redis.flushdb()
        except:
            # If using mocks, this might fail
            pass
        
        return {
            'order_repo': order_repo,
            'position_repo': position_repo,
            'idempotency_manager': idempotency_manager
        }
    
    @pytest.fixture
    def recovery_manager(self, connection_manager, event_bus, repositories):
        """Create a recovery manager for testing."""
        return RecoveryManager(
            connection_manager=connection_manager,
            event_bus=event_bus,
            order_repo=repositories['order_repo'],
            position_repo=repositories['position_repo'],
            idempotency_manager=repositories['idempotency_manager']
        )
    
    @pytest.fixture
    def paper_broker(self, event_bus):
        """Create a paper trading broker for testing."""
        config = PaperTradeConfig(
            initial_balance=100000.0,
            commission_rate=0.0005,
            slippage_rate=0.0002,
            execution_delay_ms=10,  # Fast execution for testing
            simulation_mode='realtime'
        )
        
        broker = PaperTradeAdapter(event_bus)
        broker.connect(config)
        
        # Mock market data for testing
        broker._get_current_price = mock.MagicMock(return_value=150.0)
        
        return broker
    
    @pytest.fixture
    def broker_manager(self, event_bus, paper_broker, repositories):
        """Create a multi-broker manager with idempotency for testing."""
        manager = MultiBrokerManager(event_bus=event_bus)
        
        # Set the idempotency manager
        manager.idempotency_manager = repositories['idempotency_manager']
        
        # Add the paper broker
        manager.add_broker('paper', paper_broker, None, make_primary=True)
        
        return manager
    
    @pytest.fixture
    def events_received(self):
        """Create a fixture to track events for testing."""
        return []
    
    def test_idempotent_order_placement(self, broker_manager, repositories, events_received):
        """Test idempotent order placement with persistence."""
        # Set up event handling to track events
        def event_handler(event_type, event_data):
            events_received.append((event_type, event_data))
        
        broker_manager._event_bus.subscribe(EventType.ORDER_ACKNOWLEDGED, event_handler)
        
        # Place an order idempotently
        order_params = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 10,
            'order_type': 'market',
            'time_in_force': 'day'
        }
        
        # Generate a stable idempotency key for testing
        idempotency_key = f"test_order_{order_params['symbol']}_{order_params['quantity']}"
        
        # First call should place the order
        result1 = broker_manager.place_order_idempotently(
            idempotency_key, order_params
        )
        
        # Wait for order processing
        time.sleep(0.1)
        
        # Verify the order was placed
        assert result1 is not None
        assert len(events_received) > 0
        
        # Store the broker order ID for later comparison
        broker_order_id = result1
        
        # Reset event tracking
        events_received.clear()
        
        # Second call with the same idempotency key should return the same result
        # without placing a duplicate order
        result2 = broker_manager.place_order_idempotently(
            idempotency_key, order_params
        )
        
        # Wait for order processing
        time.sleep(0.1)
        
        # Verify no new order was placed
        assert result2 == broker_order_id
        assert len(events_received) == 0  # No new events
    
    def test_crash_recovery_scenario(self, event_bus, paper_broker, repositories,
                                    recovery_manager, events_received):
        """Test a complete crash recovery scenario."""
        # Set up event handling to track events
        def event_handler(event_type, event_data):
            events_received.append((event_type, event_data))
        
        event_bus.subscribe(EventType.POSITION_UPDATE, event_handler)
        event_bus.subscribe(EventType.ORDER_ACKNOWLEDGED, event_handler)
        
        # First system: Create a broker and place orders
        broker_manager1 = MultiBrokerManager(event_bus=event_bus)
        broker_manager1.idempotency_manager = repositories['idempotency_manager']
        broker_manager1.add_broker('paper', paper_broker, None, make_primary=True)
        
        # Place orders and create positions
        order_params = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 10,
            'order_type': 'market',
            'time_in_force': 'day'
        }
        
        # Generate a stable idempotency key for testing
        idempotency_key = f"recovery_test_{order_params['symbol']}_{order_params['quantity']}"
        
        # Place the order and wait for processing
        broker_manager1.place_order_idempotently(idempotency_key, order_params)
        time.sleep(0.1)
        
        # Get positions and account info before "crash"
        original_positions = paper_broker.get_positions()
        original_account = paper_broker.get_account_info()
        
        # Verify order was placed and position created
        assert len(original_positions) > 0
        assert original_positions[0]['symbol'] == 'AAPL'
        assert original_positions[0]['quantity'] == 10
        
        # Clear event tracking before recovery
        events_received.clear()
        
        # Simulate a crash by creating a new broker manager and paper broker
        new_event_bus = EventBus()
        new_event_bus.subscribe(EventType.POSITION_UPDATE, event_handler)
        new_event_bus.subscribe(EventType.ORDER_ACKNOWLEDGED, event_handler)
        
        new_paper_broker = PaperTradeAdapter(new_event_bus)
        new_paper_broker.connect(PaperTradeConfig(
            initial_balance=100000.0,  # This will be overridden by recovery
            simulation_mode='realtime'
        ))
        
        # Mock market data for the new broker
        new_paper_broker._get_current_price = mock.MagicMock(return_value=155.0)
        
        # Create a new broker manager ("restarted" system)
        broker_manager2 = MultiBrokerManager(event_bus=new_event_bus)
        broker_manager2.idempotency_manager = repositories['idempotency_manager']
        broker_manager2.add_broker('paper', new_paper_broker, None, make_primary=True)
        
        # Perform recovery
        recovery_summary = recovery_manager.recover_full_state()
        
        # Verify recovery results
        assert recovery_summary['positions_recovered'] > 0
        
        # Check that recovery events were emitted
        position_updates = [
            (event_type, data) for event_type, data in events_received
            if event_type == EventType.POSITION_UPDATE
        ]
        assert len(position_updates) > 0
        
        # Try placing the same order again after recovery
        # This should be idempotent and not create a duplicate
        result = broker_manager2.place_order_idempotently(idempotency_key, order_params)
        time.sleep(0.1)
        
        # Verify the result
        assert result is not None
        
        # Get recovered positions
        recovered_positions = new_paper_broker.get_positions()
        
        # Verify position was recovered correctly
        assert len(recovered_positions) > 0
        assert recovered_positions[0]['symbol'] == original_positions[0]['symbol']
        assert recovered_positions[0]['quantity'] == original_positions[0]['quantity']
    
    def test_event_replay_and_reconciliation(self, event_bus, paper_broker,
                                           repositories, recovery_manager):
        """Test event replay and state reconciliation during recovery."""
        # Set up the system with some initial state
        paper_broker._get_current_price = mock.MagicMock(return_value=150.0)
        
        # Create a transaction history in the order repository
        order_repo = repositories['order_repo']
        position_repo = repositories['position_repo']
        
        # Create and save a test order
        test_order = {
            'internal_id': 'test-order-1',
            'broker': 'paper',
            'symbol': 'MSFT',
            'quantity': 15.0,
            'side': 'buy',
            'order_type': 'limit',
            'time_in_force': 'day',
            'status': 'filled',
            'limit_price': 240.0,
            'created_at': datetime.now() - timedelta(hours=2),
            'updated_at': datetime.now() - timedelta(hours=1),
            'filled_quantity': 15.0,
            'avg_fill_price': 239.5,
            'broker_order_id': 'paper-order-123'
        }
        order_repo.save_order(test_order)
        
        # Create and save a test position
        test_position = {
            'symbol': 'MSFT',
            'quantity': 15.0,
            'avg_cost': 239.5,
            'broker': 'paper',
            'last_updated': datetime.now() - timedelta(hours=1),
            'unrealized_pnl': 150.0,
            'position_id': 'paper:MSFT',
            'strategy': 'momentum'
        }
        position_repo.save_position(test_position)
        
        # Perform recovery
        events_received = []
        
        def event_handler(event_type, event_data):
            events_received.append((event_type, event_data))
        
        event_bus.subscribe(EventType.POSITION_UPDATE, event_handler)
        
        # Perform recovery
        recovery_summary = recovery_manager.recover_full_state()
        
        # Verify recovery results
        assert recovery_summary['positions_recovered'] > 0
        assert 'paper:MSFT' in recovery_summary['position_symbols']
        
        # Verify events were emitted
        position_events = [
            event_data for event_type, event_data in events_received
            if event_type == EventType.POSITION_UPDATE
        ]
        assert len(position_events) > 0
        
        # Find the MSFT position event
        msft_position_event = next(
            (e for e in position_events if e['symbol'] == 'MSFT'), None
        )
        assert msft_position_event is not None
        assert msft_position_event['quantity'] == 15.0
        assert msft_position_event['broker'] == 'paper'
        assert msft_position_event['strategy'] == 'momentum'
        assert msft_position_event['metadata']['recovered'] is True
