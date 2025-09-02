"""
Unit tests for idempotency functionality.

Tests the idempotency mechanism that prevents duplicate operations during crash recovery.
"""

import pytest
import json
from unittest import mock
from datetime import datetime, timedelta

from trading_bot.persistence.idempotency import IdempotencyManager, idempotent
from trading_bot.persistence.connection_manager import ConnectionManager


class TestIdempotencyManager:
    """Test suite for IdempotencyManager."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        return mock.MagicMock(spec=ConnectionManager)
    
    @pytest.fixture
    def mock_mongo_repo(self):
        """Create a mock MongoDB repository."""
        mongo_repo = mock.MagicMock()
        # Set up default behaviors
        mongo_repo.find_by_query.return_value = []
        return mongo_repo
    
    @pytest.fixture
    def idempotency_manager(self, mock_connection_manager, mock_mongo_repo):
        """Create an IdempotencyManager with mocked dependencies."""
        manager = IdempotencyManager(mock_connection_manager)
        manager.mongo_repo = mock_mongo_repo
        # Start with empty cache
        manager.cache = {}
        return manager
    
    def test_register_operation(self, idempotency_manager, mock_mongo_repo):
        """Test registering a new operation."""
        # Define operation parameters
        operation_type = "place_order"
        broker = "alpaca"
        params = {"symbol": "AAPL", "quantity": 10, "side": "buy"}
        
        # Call the method
        idempotency_key = idempotency_manager.register_operation(
            operation_type, broker, params
        )
        
        # Verify the key was generated
        assert idempotency_key is not None
        assert isinstance(idempotency_key, str)
        
        # Verify the record was saved to MongoDB
        mock_mongo_repo.save.assert_called_once()
        saved_record = mock_mongo_repo.save.call_args[0][0]
        
        # Verify the saved record
        assert saved_record['idempotency_key'] == idempotency_key
        assert saved_record['operation_type'] == operation_type
        assert saved_record['broker'] == broker
        assert saved_record['params'] == params
        assert saved_record['result'] is None
        
        # Verify the record was cached
        assert idempotency_key in idempotency_manager.cache
        assert idempotency_manager.cache[idempotency_key] == saved_record
    
    def test_get_operation(self, idempotency_manager, mock_mongo_repo):
        """Test retrieving an operation."""
        # Create a test record
        test_record = {
            'idempotency_key': 'test-key',
            'operation_type': 'place_order',
            'broker': 'alpaca',
            'params': {'symbol': 'AAPL', 'quantity': 10},
            'created_at': datetime.now().isoformat(),
            'result': None
        }
        
        # Set up cache hit scenario
        idempotency_manager.cache['test-key'] = test_record
        
        # Call the method
        operation = idempotency_manager.get_operation('test-key')
        
        # Verify the result
        assert operation == test_record
        
        # Verify MongoDB was not queried (cache hit)
        mock_mongo_repo.find_by_query.assert_not_called()
        
        # Test cache miss scenario
        idempotency_manager.cache = {}  # Clear cache
        mock_mongo_repo.find_by_query.return_value = [test_record]
        
        # Call the method again
        operation = idempotency_manager.get_operation('test-key')
        
        # Verify the result
        assert operation == test_record
        
        # Verify MongoDB was queried (cache miss)
        mock_mongo_repo.find_by_query.assert_called_once_with(
            {'idempotency_key': 'test-key'}
        )
        
        # Verify the result was cached
        assert 'test-key' in idempotency_manager.cache
    
    def test_record_result(self, idempotency_manager, mock_mongo_repo):
        """Test recording the result of an operation."""
        # Create a test record
        test_record = {
            'idempotency_key': 'test-key',
            'operation_type': 'place_order',
            'broker': 'alpaca',
            'params': {'symbol': 'AAPL', 'quantity': 10},
            'created_at': datetime.now().isoformat(),
            'result': None,
            '_id': 'mongo-id-123'
        }
        
        # Add to cache
        idempotency_manager.cache['test-key'] = test_record
        
        # Define a result
        result = {'order_id': 'broker-order-123', 'status': 'filled'}
        
        # Call the method
        success = idempotency_manager.record_result('test-key', result)
        
        # Verify the result
        assert success is True
        
        # Verify MongoDB was updated
        mock_mongo_repo.update.assert_called_once()
        record_id, updated_record = mock_mongo_repo.update.call_args[0]
        
        # Verify the updated record
        assert record_id == 'mongo-id-123'
        assert updated_record['result'] == result
        
        # Verify the cache was updated
        assert idempotency_manager.cache['test-key']['result'] == result
    
    def test_find_pending_operations(self, idempotency_manager, mock_mongo_repo):
        """Test finding pending operations."""
        # Create test records
        pending_op = {
            'idempotency_key': 'pending-key',
            'operation_type': 'place_order',
            'broker': 'alpaca',
            'params': {'symbol': 'AAPL', 'quantity': 10},
            'created_at': datetime.now().isoformat(),
            'result': None
        }
        
        completed_op = {
            'idempotency_key': 'completed-key',
            'operation_type': 'place_order',
            'broker': 'alpaca',
            'params': {'symbol': 'MSFT', 'quantity': 5},
            'created_at': datetime.now().isoformat(),
            'result': {'order_id': 'order-123'}
        }
        
        # Set up mock behavior
        mock_mongo_repo.find_by_query.return_value = [pending_op, completed_op]
        
        # Call the method without filters
        pending_ops = idempotency_manager.find_pending_operations()
        
        # Verify the query
        mock_mongo_repo.find_by_query.assert_called_once()
        query = mock_mongo_repo.find_by_query.call_args[0][0]
        assert query['result'] is None
        
        # Verify the result
        assert len(pending_ops) == 1
        assert pending_ops[0]['idempotency_key'] == 'pending-key'
        
        # Test with operation_type filter
        mock_mongo_repo.find_by_query.reset_mock()
        mock_mongo_repo.find_by_query.return_value = [pending_op]
        
        # Call the method with filter
        pending_ops = idempotency_manager.find_pending_operations(
            operation_type='place_order', broker='alpaca'
        )
        
        # Verify the query
        mock_mongo_repo.find_by_query.assert_called_once()
        query = mock_mongo_repo.find_by_query.call_args[0][0]
        assert query['result'] is None
        assert query['operation_type'] == 'place_order'
        assert query['broker'] == 'alpaca'
        
        # Verify the result
        assert len(pending_ops) == 1
        assert pending_ops[0]['idempotency_key'] == 'pending-key'


class TestIdempotentDecorator:
    """Test suite for @idempotent decorator."""
    
    class TestBrokerAdapter:
        """Test class for demonstrating the @idempotent decorator."""
        
        def __init__(self):
            self.broker_id = 'test-broker'
            self.idempotency_manager = mock.MagicMock(spec=IdempotencyManager)
            self.called_count = 0
        
        @idempotent('place_order', ['symbol', 'quantity', 'side'])
        def place_order(self, symbol, quantity, side, idempotency_key=None):
            """Place an order with idempotency."""
            self.called_count += 1
            return {'order_id': f'order-{symbol}-{quantity}', 'status': 'new'}
    
    @pytest.fixture
    def test_adapter(self):
        """Create a test adapter with the decorated method."""
        return self.TestBrokerAdapter()
    
    def test_idempotent_first_call(self, test_adapter):
        """Test the first call to an idempotent method."""
        # Set up mock behavior for a new operation
        test_adapter.idempotency_manager.get_operation.return_value = None
        test_adapter.idempotency_manager.register_operation.return_value = 'new-key'
        
        # Call the decorated method
        result = test_adapter.place_order('AAPL', 10, 'buy')
        
        # Verify the operation was registered
        test_adapter.idempotency_manager.register_operation.assert_called_once_with(
            'place_order', 'test-broker', {'symbol': 'AAPL', 'quantity': 10, 'side': 'buy'}
        )
        
        # Verify the result was recorded
        test_adapter.idempotency_manager.record_result.assert_called_once_with(
            'new-key', result
        )
        
        # Verify the original method was called
        assert test_adapter.called_count == 1
        
        # Verify the result
        assert result['order_id'] == 'order-AAPL-10'
        assert result['status'] == 'new'
    
    def test_idempotent_repeat_call(self, test_adapter):
        """Test a repeated call to an idempotent method with the same key."""
        # Set up mock behavior for an existing operation with a result
        existing_operation = {
            'idempotency_key': 'existing-key',
            'operation_type': 'place_order',
            'broker': 'test-broker',
            'params': {'symbol': 'AAPL', 'quantity': 10, 'side': 'buy'},
            'result': {'order_id': 'cached-order-123', 'status': 'filled'}
        }
        test_adapter.idempotency_manager.get_operation.return_value = existing_operation
        
        # Call the decorated method with the same idempotency key
        result = test_adapter.place_order(
            'AAPL', 10, 'buy', idempotency_key='existing-key'
        )
        
        # Verify the operation was looked up
        test_adapter.idempotency_manager.get_operation.assert_called_once_with(
            'existing-key'
        )
        
        # Verify no new operation was registered
        test_adapter.idempotency_manager.register_operation.assert_not_called()
        
        # Verify the original method was NOT called
        assert test_adapter.called_count == 0
        
        # Verify the result matches the cached result
        assert result['order_id'] == 'cached-order-123'
        assert result['status'] == 'filled'
    
    def test_idempotent_no_manager(self, test_adapter):
        """Test behavior when no idempotency manager is available."""
        # Remove the idempotency manager
        test_adapter.idempotency_manager = None
        
        # Call the decorated method
        result = test_adapter.place_order('AAPL', 10, 'buy')
        
        # Verify the original method was called
        assert test_adapter.called_count == 1
        
        # Verify the result
        assert result['order_id'] == 'order-AAPL-10'
        assert result['status'] == 'new'
