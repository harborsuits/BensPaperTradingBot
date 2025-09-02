"""
Unit tests for broker adapters.

Tests the AlpacaAdapter and PaperTradeAdapter implementations of the BrokerInterface.
"""

import pytest
import pandas as pd
from unittest import mock
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import EventType, OrderStatus
from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.brokers.alpaca.adapter import AlpacaAdapter
from trading_bot.brokers.paper.adapter import PaperTradeAdapter, PaperTradeConfig


class TestAlpacaAdapter:
    """Test suite for AlpacaAdapter."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a mock event bus."""
        return mock.MagicMock(spec=EventBus)
    
    @pytest.fixture
    def mock_alpaca_api(self):
        """Create a mock Alpaca API."""
        mock_api = mock.MagicMock()
        # Mock common methods
        mock_api.get_account.return_value = {
            'buying_power': '100000',
            'cash': '50000',
            'equity': '150000',
            'id': 'test-account'
        }
        mock_api.list_positions.return_value = []
        mock_api.list_orders.return_value = []
        return mock_api
    
    @pytest.fixture
    def adapter(self, event_bus, mock_alpaca_api):
        """Create an AlpacaAdapter with mocked dependencies."""
        adapter = AlpacaAdapter(event_bus)
        adapter._api = mock_alpaca_api  # Inject the mock API
        return adapter
    
    def test_get_account_info(self, adapter, mock_alpaca_api):
        """Test retrieving account information."""
        # Set up mock response
        mock_alpaca_api.get_account.return_value = {
            'account_number': 'ALPACA123',
            'buying_power': '100000.0',
            'cash': '50000.0',
            'equity': '150000.0',
            'initial_margin': '0.0',
            'maintenance_margin': '0.0',
            'id': 'test-account'
        }
        
        # Call the method
        account_info = adapter.get_account_info()
        
        # Verify the result
        assert account_info is not None
        assert account_info.get('account_id') == 'test-account'
        assert float(account_info.get('buying_power')) == 100000.0
        assert float(account_info.get('cash')) == 50000.0
        assert float(account_info.get('equity')) == 150000.0
    
    def test_get_positions(self, adapter, mock_alpaca_api):
        """Test retrieving positions."""
        # Set up mock response
        mock_alpaca_api.list_positions.return_value = [
            {
                'symbol': 'AAPL',
                'qty': '10',
                'avg_entry_price': '150.0',
                'current_price': '155.0',
                'unrealized_pl': '50.0',
                'side': 'long'
            },
            {
                'symbol': 'MSFT',
                'qty': '5',
                'avg_entry_price': '300.0',
                'current_price': '305.0',
                'unrealized_pl': '25.0',
                'side': 'long'
            }
        ]
        
        # Call the method
        positions = adapter.get_positions()
        
        # Verify the result
        assert len(positions) == 2
        assert positions[0]['symbol'] == 'AAPL'
        assert float(positions[0]['quantity']) == 10
        assert float(positions[0]['avg_cost']) == 150.0
        assert float(positions[0]['current_price']) == 155.0
        
        assert positions[1]['symbol'] == 'MSFT'
        assert float(positions[1]['quantity']) == 5
        assert float(positions[1]['avg_cost']) == 300.0
        assert float(positions[1]['current_price']) == 305.0
    
    def test_place_equity_order(self, adapter, mock_alpaca_api, event_bus):
        """Test placing an equity order."""
        # Set up mock response
        mock_alpaca_api.submit_order.return_value = {
            'id': 'test-order-id',
            'client_order_id': 'client-123',
            'symbol': 'AAPL',
            'qty': '10',
            'type': 'market',
            'side': 'buy',
            'status': 'new'
        }
        
        # Call the method
        order_id = adapter.place_equity_order(
            symbol='AAPL',
            side='buy',
            quantity=10,
            order_type='market',
            time_in_force='day',
            price=None,
            stop_price=None,
            client_order_id='client-123'
        )
        
        # Verify the API was called correctly
        mock_alpaca_api.submit_order.assert_called_once_with(
            symbol='AAPL',
            qty=10,
            side='buy',
            type='market',
            time_in_force='day',
            limit_price=None,
            stop_price=None,
            client_order_id='client-123'
        )
        
        # Verify the result
        assert order_id == 'test-order-id'
        
        # Verify an event was emitted
        event_bus.emit.assert_called_with(
            EventType.ORDER_ACKNOWLEDGED,
            mock.ANY  # We can't easily check the exact event data
        )
    
    def test_cancel_order(self, adapter, mock_alpaca_api):
        """Test cancelling an order."""
        # Set up mock response
        mock_alpaca_api.cancel_order.return_value = {'id': 'test-order-id', 'status': 'canceled'}
        
        # Call the method
        result = adapter.cancel_order('test-order-id')
        
        # Verify the API was called correctly
        mock_alpaca_api.cancel_order.assert_called_once_with('test-order-id')
        
        # Verify the result
        assert result is True
    
    def test_get_equity_quote(self, adapter, mock_alpaca_api):
        """Test getting an equity quote."""
        # Set up mock response
        mock_alpaca_api.get_latest_quote.return_value = {
            'symbol': 'AAPL', 
            'ask_price': 155.25,
            'ask_size': 100,
            'bid_price': 155.20,
            'bid_size': 200,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Call the method
        quote = adapter.get_equity_quote('AAPL')
        
        # Verify the API was called correctly
        mock_alpaca_api.get_latest_quote.assert_called_once_with('AAPL')
        
        # Verify the result
        assert quote['symbol'] == 'AAPL'
        assert quote['ask'] == 155.25
        assert quote['bid'] == 155.20


class TestPaperTradeAdapter:
    """Test suite for PaperTradeAdapter."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a mock event bus."""
        return mock.MagicMock(spec=EventBus)
    
    @pytest.fixture
    def paper_config(self):
        """Create a PaperTradeConfig for testing."""
        return PaperTradeConfig(
            initial_balance=100000.0,
            commission_rate=0.0005,
            slippage_rate=0.0002,
            execution_delay_ms=100,
            simulation_mode='realtime'
        )
    
    @pytest.fixture
    def adapter(self, event_bus, paper_config):
        """Create a PaperTradeAdapter with test configuration."""
        adapter = PaperTradeAdapter(event_bus)
        adapter.connect(paper_config)
        return adapter
    
    def test_initial_account_state(self, adapter):
        """Test the initial account state."""
        account_info = adapter.get_account_info()
        
        # Verify the result
        assert account_info is not None
        assert account_info.get('account_id') == 'paper_account'
        assert account_info.get('buying_power') == 100000.0
        assert account_info.get('cash') == 100000.0
        assert account_info.get('equity') == 100000.0
    
    def test_place_and_execute_order(self, adapter, event_bus):
        """Test placing and executing an order in the paper trading environment."""
        # Set up a mock quote
        adapter._get_current_price = mock.MagicMock(return_value=150.0)
        
        # Place an order
        order_id = adapter.place_equity_order(
            symbol='AAPL',
            side='buy',
            quantity=10,
            order_type='market',
            time_in_force='day'
        )
        
        # Verify the order was created
        assert order_id is not None
        assert len(adapter._orders) == 1
        assert adapter._orders[order_id]['symbol'] == 'AAPL'
        assert adapter._orders[order_id]['quantity'] == 10
        assert adapter._orders[order_id]['side'] == 'buy'
        
        # Simulate order execution
        adapter._process_open_orders()
        
        # Verify the position was created
        positions = adapter.get_positions()
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'AAPL'
        assert positions[0]['quantity'] == 10
        
        # Verify the account was updated
        account_info = adapter.get_account_info()
        expected_cost = 10 * 150.0
        expected_commission = expected_cost * 0.0005
        expected_remaining = 100000.0 - expected_cost - expected_commission
        
        assert account_info.get('cash') == pytest.approx(expected_remaining, abs=0.01)
    
    def test_backtest_mode(self, event_bus):
        """Test the adapter in backtest mode."""
        # Create config for backtest mode
        config = PaperTradeConfig(
            initial_balance=100000.0,
            simulation_mode='backtest'
        )
        
        # Create adapter in backtest mode
        adapter = PaperTradeAdapter(event_bus)
        adapter.connect(config)
        
        # Load some historical data
        historical_data = {
            'AAPL': pd.DataFrame({
                'open': [150.0, 151.0, 152.0, 153.0],
                'high': [155.0, 156.0, 157.0, 158.0],
                'low': [149.0, 150.0, 151.0, 152.0],
                'close': [154.0, 155.0, 156.0, 157.0],
                'volume': [1000000, 1100000, 1200000, 1300000]
            }, index=pd.date_range(start='2025-01-01', periods=4, freq='D'))
        }
        
        adapter.load_historical_data(historical_data)
        
        # Set the backtest time to the first timestamp
        adapter.set_backtest_time(pd.Timestamp('2025-01-01'))
        
        # Place an order
        order_id = adapter.place_equity_order(
            symbol='AAPL',
            side='buy',
            quantity=10,
            order_type='market',
            time_in_force='day'
        )
        
        # Process open orders at the current time
        adapter._process_open_orders()
        
        # Verify the position was created using the right price
        positions = adapter.get_positions()
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'AAPL'
        assert positions[0]['quantity'] == 10
        assert positions[0]['avg_cost'] == pytest.approx(154.0, abs=0.1)  # Using close price
    
    def test_crash_recovery(self, adapter):
        """Test the adapter's ability to recover state after a crash."""
        # Set up a position
        adapter._get_current_price = mock.MagicMock(return_value=150.0)
        adapter.place_equity_order(
            symbol='AAPL',
            side='buy',
            quantity=10,
            order_type='market',
            time_in_force='day'
        )
        adapter._process_open_orders()
        
        # Get the current state
        original_positions = adapter.get_positions()
        original_account = adapter.get_account_info()
        
        # Create a new adapter (simulating a restart)
        new_adapter = PaperTradeAdapter(adapter._event_bus)
        new_adapter.connect(adapter._config)
        
        # Load the state from the first adapter
        positions_data = [
            {
                'symbol': pos['symbol'],
                'quantity': pos['quantity'],
                'avg_cost': pos['avg_cost'],
                'open_date': datetime.now().isoformat()
            }
            for pos in original_positions
        ]
        
        account_data = {
            'cash': original_account['cash'],
            'equity': original_account['equity']
        }
        
        # Simulate recovery
        new_adapter.load_state(positions_data, account_data)
        
        # Verify the state was recovered
        recovered_positions = new_adapter.get_positions()
        assert len(recovered_positions) == len(original_positions)
        assert recovered_positions[0]['symbol'] == original_positions[0]['symbol']
        assert recovered_positions[0]['quantity'] == original_positions[0]['quantity']
        assert recovered_positions[0]['avg_cost'] == original_positions[0]['avg_cost']
        
        recovered_account = new_adapter.get_account_info()
        assert recovered_account['cash'] == original_account['cash']
        assert recovered_account['equity'] == original_account['equity']
