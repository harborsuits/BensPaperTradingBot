"""
Unit tests for the RecoveryManager.

Tests the recovery functionality that handles crash recovery and state reconciliation.
"""

import pytest
import json
from unittest import mock
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import EventType, OrderStatus
from trading_bot.persistence.recovery_manager import RecoveryManager
from trading_bot.persistence.order_repository import OrderRepository, OrderModel
from trading_bot.persistence.position_repository import PositionRepository, PositionModel
from trading_bot.persistence.fill_repository import FillRepository
from trading_bot.persistence.pnl_repository import PnLRepository
from trading_bot.persistence.idempotency import IdempotencyManager
from trading_bot.persistence.connection_manager import ConnectionManager


class TestRecoveryManager:
    """Test suite for RecoveryManager."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        return mock.MagicMock(spec=ConnectionManager)
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return mock.MagicMock(spec=EventBus)
    
    @pytest.fixture
    def mock_order_repo(self):
        """Create a mock order repository."""
        repo = mock.MagicMock(spec=OrderRepository)
        # Set up default behaviors
        repo.fetch_open_orders.return_value = []
        return repo
    
    @pytest.fixture
    def mock_position_repo(self):
        """Create a mock position repository."""
        repo = mock.MagicMock(spec=PositionRepository)
        # Set up default behaviors
        repo.find_all_positions.return_value = []
        return repo
    
    @pytest.fixture
    def mock_fill_repo(self):
        """Create a mock fill repository."""
        return mock.MagicMock(spec=FillRepository)
    
    @pytest.fixture
    def mock_pnl_repo(self):
        """Create a mock PnL repository."""
        repo = mock.MagicMock(spec=PnLRepository)
        # Set up default behaviors
        repo.get_latest_snapshot.return_value = None
        return repo
    
    @pytest.fixture
    def mock_idempotency_manager(self):
        """Create a mock idempotency manager."""
        manager = mock.MagicMock(spec=IdempotencyManager)
        # Set up default behaviors
        manager.find_pending_operations.return_value = []
        return manager
    
    @pytest.fixture
    def recovery_manager(self, mock_connection_manager, mock_event_bus, mock_order_repo,
                        mock_position_repo, mock_fill_repo, mock_pnl_repo, 
                        mock_idempotency_manager):
        """Create a RecoveryManager with mocked dependencies."""
        return RecoveryManager(
            connection_manager=mock_connection_manager,
            event_bus=mock_event_bus,
            order_repo=mock_order_repo,
            position_repo=mock_position_repo,
            fill_repo=mock_fill_repo,
            pnl_repo=mock_pnl_repo,
            idempotency_manager=mock_idempotency_manager
        )
    
    def test_recover_open_orders(self, recovery_manager, mock_order_repo, mock_event_bus):
        """Test recovering open orders."""
        # Create sample order data
        open_orders = [
            OrderModel(
                internal_id="order1",
                broker="alpaca",
                symbol="AAPL",
                quantity=10.0,
                side="buy",
                order_type="limit",
                time_in_force="day",
                status="acknowledged",
                broker_order_id="broker-order-1",
                limit_price=150.0,
                created_at=datetime.now() - timedelta(hours=1),
                updated_at=datetime.now() - timedelta(minutes=30)
            ),
            OrderModel(
                internal_id="order2",
                broker="tradier",
                symbol="MSFT",
                quantity=5.0,
                side="sell",
                order_type="market",
                time_in_force="day",
                status="new",
                broker_order_id="broker-order-2",
                created_at=datetime.now() - timedelta(minutes=15),
                updated_at=datetime.now() - timedelta(minutes=15)
            )
        ]
        
        # Set up mock to return sample orders
        mock_order_repo.fetch_open_orders.return_value = open_orders
        
        # Call the recovery method
        recovered_orders = recovery_manager.recover_open_orders()
        
        # Verify results
        assert len(recovered_orders) == 2
        
        # Verify events were emitted for each order
        assert mock_event_bus.emit.call_count == 2
        
        # Verify the event data for the first call
        event_type, event_data = mock_event_bus.emit.call_args_list[0][0]
        assert event_type == EventType.ORDER_ACKNOWLEDGED
        assert event_data['order_id'] == "order1"
        assert event_data['symbol'] == "AAPL"
        assert event_data['broker'] == "alpaca"
        assert event_data['metadata']['recovered'] is True
        
        # Verify the event data for the second call
        event_type, event_data = mock_event_bus.emit.call_args_list[1][0]
        assert event_type == EventType.ORDER_ACKNOWLEDGED
        assert event_data['order_id'] == "order2"
        assert event_data['symbol'] == "MSFT"
        assert event_data['broker'] == "tradier"
        assert event_data['metadata']['recovered'] is True
    
    def test_recover_positions(self, recovery_manager, mock_position_repo, mock_event_bus):
        """Test recovering positions."""
        # Create sample position data
        positions = [
            PositionModel(
                symbol="AAPL",
                quantity=10.0,
                avg_cost=150.0,
                broker="alpaca",
                last_updated=datetime.now() - timedelta(hours=2),
                unrealized_pnl=500.0,
                strategy="momentum",
                position_id="alpaca:AAPL"
            ),
            PositionModel(
                symbol="MSFT",
                quantity=-5.0,  # Short position
                avg_cost=300.0,
                broker="tradier",
                last_updated=datetime.now() - timedelta(hours=1),
                unrealized_pnl=-100.0,
                strategy="mean_reversion",
                position_id="tradier:MSFT"
            )
        ]
        
        # Set up mock to return sample positions
        mock_position_repo.find_all_positions.return_value = positions
        
        # Call the recovery method
        recovered_positions = recovery_manager.recover_positions()
        
        # Verify results
        assert len(recovered_positions) == 2
        
        # Verify events were emitted for each position
        assert mock_event_bus.emit.call_count == 2
        
        # Verify the event data for the first call
        event_type, event_data = mock_event_bus.emit.call_args_list[0][0]
        assert event_type == EventType.POSITION_UPDATE
        assert event_data['symbol'] == "AAPL"
        assert event_data['quantity'] == 10.0
        assert event_data['avg_cost'] == 150.0
        assert event_data['broker'] == "alpaca"
        assert event_data['strategy'] == "momentum"
        assert event_data['metadata']['recovered'] is True
        
        # Verify the event data for the second call
        event_type, event_data = mock_event_bus.emit.call_args_list[1][0]
        assert event_type == EventType.POSITION_UPDATE
        assert event_data['symbol'] == "MSFT"
        assert event_data['quantity'] == -5.0
        assert event_data['avg_cost'] == 300.0
        assert event_data['broker'] == "tradier"
        assert event_data['strategy'] == "mean_reversion"
        assert event_data['metadata']['recovered'] is True
    
    def test_recover_latest_pnl(self, recovery_manager, mock_pnl_repo, mock_event_bus):
        """Test recovering latest P&L data."""
        # Create sample P&L data
        latest_pnl = mock.MagicMock()
        latest_pnl.total_equity = 120000.0
        latest_pnl.unrealized_pnl = 3000.0
        latest_pnl.realized_pnl = 2000.0
        latest_pnl.cash_balance = 115000.0
        latest_pnl.broker = "alpaca"
        latest_pnl.timestamp = datetime.now() - timedelta(minutes=5)
        latest_pnl.drawdown = 1500.0
        latest_pnl.drawdown_pct = 1.25
        
        # Set up mock to return sample P&L
        mock_pnl_repo.get_latest_snapshot.return_value = latest_pnl
        
        # Call the recovery method
        recovered_pnl = recovery_manager.recover_latest_pnl()
        
        # Verify results
        assert recovered_pnl is not None
        assert recovered_pnl['total_equity'] == 120000.0
        assert recovered_pnl['unrealized_pnl'] == 3000.0
        assert recovered_pnl['realized_pnl'] == 2000.0
        
        # Verify event was emitted
        mock_event_bus.emit.assert_called_once()
        
        # Verify the event data
        event_type, event_data = mock_event_bus.emit.call_args[0]
        assert event_type == EventType.PORTFOLIO_EQUITY_UPDATE
        assert event_data['total_equity'] == 120000.0
        assert event_data['unrealized_pnl'] == 3000.0
        assert event_data['realized_pnl'] == 2000.0
        assert event_data['cash_balance'] == 115000.0
        assert event_data['broker'] == "alpaca"
        assert event_data['metadata']['recovered'] is True
    
    def test_recover_idempotency_state(self, recovery_manager, mock_idempotency_manager):
        """Test recovering idempotency state."""
        # Create sample pending operations
        pending_operations = [
            {
                'idempotency_key': 'key1',
                'operation_type': 'place_order',
                'broker': 'alpaca',
                'params': {'symbol': 'AAPL', 'quantity': 10}
            },
            {
                'idempotency_key': 'key2',
                'operation_type': 'cancel_order',
                'broker': 'tradier',
                'params': {'order_id': 'order1'}
            }
        ]
        
        # Set up mock to return sample pending operations
        mock_idempotency_manager.find_pending_operations.return_value = pending_operations
        
        # Call the recovery method
        num_pending = recovery_manager.recover_idempotency_state()
        
        # Verify results
        assert num_pending == 2
        
        # Verify the cache initialization was called
        mock_idempotency_manager._init_cache.assert_called_once()
    
    def test_recover_full_state(self, recovery_manager):
        """Test recovering full state."""
        # Set up mocks for individual recovery methods
        recovery_manager.recover_open_orders = mock.MagicMock(return_value=[
            OrderModel(internal_id="order1", broker="alpaca", symbol="AAPL", 
                      quantity=10.0, side="buy", order_type="limit", 
                      time_in_force="day", status="acknowledged")
        ])
        
        recovery_manager.recover_positions = mock.MagicMock(return_value=[
            PositionModel(symbol="AAPL", quantity=10.0, avg_cost=150.0, 
                         broker="alpaca", position_id="alpaca:AAPL")
        ])
        
        recovery_manager.recover_latest_pnl = mock.MagicMock(return_value={
            'total_equity': 120000.0,
            'unrealized_pnl': 3000.0,
            'realized_pnl': 2000.0
        })
        
        recovery_manager.recover_idempotency_state = mock.MagicMock(return_value=2)
        
        # Call the full recovery method
        recovery_summary = recovery_manager.recover_full_state()
        
        # Verify the individual recovery methods were called
        recovery_manager.recover_open_orders.assert_called_once()
        recovery_manager.recover_positions.assert_called_once()
        recovery_manager.recover_latest_pnl.assert_called_once()
        
        # Verify the recovery summary
        assert recovery_summary['open_orders_recovered'] == 1
        assert recovery_summary['positions_recovered'] == 1
        assert recovery_summary['pnl_recovered'] is True
        assert len(recovery_summary['order_ids']) == 1
        assert recovery_summary['order_ids'][0] == "order1"
