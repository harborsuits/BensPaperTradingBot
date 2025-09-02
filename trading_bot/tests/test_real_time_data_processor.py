#!/usr/bin/env python3
"""
Unit tests for the real-time data processor module.

These tests validate the functionality of the data sources, processors, and manager
components of the real-time data processing pipeline.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import queue
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components to test
from trading_bot.data.real_time_data_processor import (
    MarketDataSource,
    AlpacaDataSource,
    IBDataSource,
    DataProcessor,
    RealTimeDataManager
)

class TestMarketDataSource(unittest.TestCase):
    """Tests for the abstract MarketDataSource class"""
    
    def setUp(self):
        # Create a concrete implementation for testing
        class ConcreteDataSource(MarketDataSource):
            def connect(self):
                return True
                
            def disconnect(self):
                pass
                
            def start_streaming(self):
                self.is_running = True
                
            def stop_streaming(self):
                self.is_running = False
        
        self.symbols = ["SPY", "QQQ"]
        self.data_source = ConcreteDataSource(self.symbols)
    
    def test_initialization(self):
        """Test proper initialization of the data source"""
        self.assertEqual(self.data_source.symbols, self.symbols)
        self.assertFalse(self.data_source.is_running)
        self.assertIsNone(self.data_source.last_update_time)
        self.assertEqual(self.data_source.reconnect_attempts, 0)
    
    def test_callback_registration(self):
        """Test callback registration and unregistration"""
        callback = Mock()
        
        # Register callback
        self.data_source.register_callback(callback)
        self.assertIn(callback, self.data_source.callbacks)
        
        # Register same callback again (should not duplicate)
        self.data_source.register_callback(callback)
        self.assertEqual(len(self.data_source.callbacks), 1)
        
        # Unregister callback
        self.data_source.unregister_callback(callback)
        self.assertNotIn(callback, self.data_source.callbacks)
    
    def test_process_data(self):
        """Test processing data and calling callbacks"""
        callback1 = Mock()
        callback2 = Mock()
        
        self.data_source.register_callback(callback1)
        self.data_source.register_callback(callback2)
        
        test_data = {"symbol": "SPY", "price": 400.0}
        self.data_source.process_data(test_data)
        
        # Verify callbacks were called with the data
        callback1.assert_called_once_with(test_data)
        callback2.assert_called_once_with(test_data)
        
        # Verify last update time was set
        self.assertIsNotNone(self.data_source.last_update_time)
    
    def test_is_connected(self):
        """Test connection status checking"""
        # Initially not connected
        self.assertFalse(self.data_source.is_connected())
        
        # Set running but no last update
        self.data_source.is_running = True
        self.assertFalse(self.data_source.is_connected())
        
        # Set last update to recent time
        self.data_source.last_update_time = datetime.now()
        self.assertTrue(self.data_source.is_connected())
        
        # Set last update to old time
        self.data_source.last_update_time = datetime.now() - timedelta(minutes=2)
        self.assertFalse(self.data_source.is_connected())

class TestAlpacaDataSource(unittest.TestCase):
    """Tests for the AlpacaDataSource class"""
    
    def setUp(self):
        self.symbols = ["SPY", "QQQ"]
        self.config = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "base_url": "https://test.alpaca.markets",
            "data_url": "wss://test.data.alpaca.markets/v2"
        }
        
        # Create the data source with patched websocket
        with patch('websocket.WebSocketApp'):
            self.data_source = AlpacaDataSource(self.symbols, self.config)
    
    def test_initialization(self):
        """Test proper initialization of the Alpaca data source"""
        self.assertEqual(self.data_source.symbols, self.symbols)
        self.assertEqual(self.data_source.api_key, "test_key")
        self.assertEqual(self.data_source.api_secret, "test_secret")
        self.assertFalse(self.data_source.ws_connected)
    
    def test_initialization_missing_credentials(self):
        """Test initialization without required credentials"""
        with self.assertRaises(ValueError):
            AlpacaDataSource(self.symbols, {})
    
    @patch('websocket.WebSocketApp')
    def test_connect(self, mock_websocket):
        """Test connection to Alpaca WebSocket API"""
        result = self.data_source.connect()
        self.assertTrue(result)
        mock_websocket.assert_called_once()
    
    @patch('websocket.WebSocketApp')
    def test_disconnect(self, mock_websocket):
        """Test disconnection from Alpaca WebSocket API"""
        # Setup mock WebSocket
        mock_ws = MagicMock()
        self.data_source.ws = mock_ws
        self.data_source.ws_connected = True
        
        # Mock thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        self.data_source.ws_thread = mock_thread
        
        # Test disconnect
        self.data_source.disconnect()
        
        mock_ws.close.assert_called_once()
        self.assertFalse(self.data_source.ws_connected)
        mock_thread.join.assert_called_once()
    
    @patch('requests.get')
    def test_get_latest_bars(self, mock_get):
        """Test retrieving latest OHLCV bars from Alpaca"""
        # Mock response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'bars': {
                'SPY': [
                    {
                        't': '2023-06-01T09:30:00Z',
                        'o': 400.0,
                        'h': 402.0,
                        'l': 399.0,
                        'c': 401.0,
                        'v': 10000
                    }
                ]
            }
        }
        mock_get.return_value = mock_resp
        
        # Call the method
        result = self.data_source.get_latest_bars()
        
        # Verify the result contains the expected data
        self.assertIn('SPY', result)
        self.assertEqual(len(result['SPY']), 1)
        self.assertEqual(result['SPY']['close'].iloc[0], 401.0)
        
        # Verify the API was called with the correct parameters
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        self.assertIn('alpaca.markets/v2/stocks/bars', called_url)

class TestDataProcessor(unittest.TestCase):
    """Tests for the DataProcessor class"""
    
    def setUp(self):
        self.processor = DataProcessor(['1min', '5min', '1day'])
    
    def test_initialization(self):
        """Test proper initialization of the data processor"""
        self.assertEqual(self.processor.timeframes, ['1min', '5min', '1day'])
        self.assertEqual(len(self.processor.bars), 3)
        self.assertEqual(len(self.processor.latest_ticks), 0)
    
    def test_callback_registration(self):
        """Test callback registration and unregistration"""
        callback = Mock()
        
        # Register callback
        self.processor.register_callback(callback)
        self.assertIn(callback, self.processor.callbacks)
        
        # Unregister callback
        self.processor.unregister_callback(callback)
        self.assertNotIn(callback, self.processor.callbacks)
    
    def test_process_tick(self):
        """Test processing a single tick"""
        # Invalid tick data
        self.processor.process_tick({})
        self.assertEqual(len(self.processor.latest_ticks), 0)
        
        # Valid tick data
        tick_data = {
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'price': 400.0,
            'volume': 100
        }
        self.processor.process_tick(tick_data)
        
        # Verify latest tick was stored
        self.assertIn('SPY', self.processor.latest_ticks)
        self.assertEqual(self.processor.latest_ticks['SPY']['price'], 400.0)
        
        # Verify bars were updated for all timeframes
        for timeframe in self.processor.timeframes:
            self.assertIn('SPY', self.processor.bars[timeframe])
            self.assertEqual(len(self.processor.bars[timeframe]['SPY']), 1)
    
    def test_update_bars(self):
        """Test updating OHLCV bars with tick data"""
        symbol = 'SPY'
        timeframe = '1min'
        timestamp = datetime.now().replace(second=0, microsecond=0)
        price = 400.0
        volume = 100
        
        # Update with first tick (creates new bar)
        self.processor._update_bars(symbol, timeframe, timestamp, price, volume)
        df = self.processor.bars[timeframe][symbol]
        self.assertEqual(len(df), 1)
        self.assertEqual(df.at[timestamp, 'open'], price)
        self.assertEqual(df.at[timestamp, 'high'], price)
        self.assertEqual(df.at[timestamp, 'low'], price)
        self.assertEqual(df.at[timestamp, 'close'], price)
        self.assertEqual(df.at[timestamp, 'volume'], volume)
        
        # Update with higher price (updates existing bar)
        higher_price = 405.0
        self.processor._update_bars(symbol, timeframe, timestamp, higher_price, volume)
        df = self.processor.bars[timeframe][symbol]
        self.assertEqual(len(df), 1)
        self.assertEqual(df.at[timestamp, 'open'], price)  # Should not change
        self.assertEqual(df.at[timestamp, 'high'], higher_price)  # Should update
        self.assertEqual(df.at[timestamp, 'low'], price)  # Should not change
        self.assertEqual(df.at[timestamp, 'close'], higher_price)  # Should update
        self.assertEqual(df.at[timestamp, 'volume'], volume * 2)  # Should accumulate
        
        # Update with lower price (updates existing bar)
        lower_price = 395.0
        self.processor._update_bars(symbol, timeframe, timestamp, lower_price, volume)
        df = self.processor.bars[timeframe][symbol]
        self.assertEqual(df.at[timestamp, 'low'], lower_price)  # Should update
        
        # Update with new timestamp (creates new bar)
        new_timestamp = timestamp + timedelta(minutes=1)
        self.processor._update_bars(symbol, timeframe, new_timestamp, price, volume)
        df = self.processor.bars[timeframe][symbol]
        self.assertEqual(len(df), 2)
    
    def test_floor_timestamp(self):
        """Test converting timestamps to timeframe boundaries"""
        now = datetime.now()
        
        # Test 1-minute timeframe
        floored = self.processor._floor_timestamp(now, '1min')
        self.assertEqual(floored.second, 0)
        self.assertEqual(floored.microsecond, 0)
        
        # Test 5-minute timeframe
        floored = self.processor._floor_timestamp(now, '5min')
        self.assertEqual(floored.minute // 5 * 5, floored.minute)
        self.assertEqual(floored.second, 0)
        self.assertEqual(floored.microsecond, 0)
        
        # Test 1-hour timeframe
        floored = self.processor._floor_timestamp(now, '1hour')
        self.assertEqual(floored.minute, 0)
        self.assertEqual(floored.second, 0)
        self.assertEqual(floored.microsecond, 0)
        
        # Test 1-day timeframe
        floored = self.processor._floor_timestamp(now, '1day')
        self.assertEqual(floored.hour, 0)
        self.assertEqual(floored.minute, 0)
        self.assertEqual(floored.second, 0)
        self.assertEqual(floored.microsecond, 0)
    
    def test_get_latest_bars(self):
        """Test retrieving latest bars for a symbol and timeframe"""
        # Test with non-existent data
        df = self.processor.get_latest_bars('AAPL', '1min')
        self.assertTrue(df.empty)
        
        # Add some data and test retrieval
        symbol = 'SPY'
        timeframe = '1min'
        
        # Create some test data
        now = datetime.now().replace(second=0, microsecond=0)
        for i in range(10):
            timestamp = now - timedelta(minutes=i)
            self.processor._update_bars(symbol, timeframe, timestamp, 400.0 + i, 100)
        
        # Get the latest bars
        df = self.processor.get_latest_bars(symbol, timeframe, 5)
        self.assertEqual(len(df), 5)
        
        # Get all bars
        df = self.processor.get_latest_bars(symbol, timeframe, 20)
        self.assertEqual(len(df), 10)
    
    def test_get_latest_tick(self):
        """Test retrieving the latest tick for a symbol"""
        # Test with non-existent data
        tick = self.processor.get_latest_tick('AAPL')
        self.assertEqual(tick, {})
        
        # Add some data and test retrieval
        symbol = 'SPY'
        tick_data = {
            'timestamp': datetime.now(),
            'price': 400.0,
            'volume': 100,
            'bid': 399.5,
            'ask': 400.5
        }
        self.processor.latest_ticks[symbol] = tick_data
        
        # Get the latest tick
        tick = self.processor.get_latest_tick(symbol)
        self.assertEqual(tick, tick_data)

class TestRealTimeDataManager(unittest.TestCase):
    """Tests for the RealTimeDataManager class"""
    
    def setUp(self):
        self.symbols = ["SPY", "QQQ"]
        
        # Mock the data sources
        self.mock_alpaca = MagicMock(spec=AlpacaDataSource)
        self.mock_ib = MagicMock(spec=IBDataSource)
        
        # Patch the imports
        self.patcher1 = patch('trading_bot.data.real_time_data_processor.AlpacaDataSource', return_value=self.mock_alpaca)
        self.patcher2 = patch('trading_bot.data.real_time_data_processor.IBDataSource', return_value=self.mock_ib)
        self.patcher3 = patch('trading_bot.data.real_time_data_processor.AdvancedMarketRegimeDetector')
        
        self.mock_alpaca_class = self.patcher1.start()
        self.mock_ib_class = self.patcher2.start()
        self.mock_regime_detector = self.patcher3.start()
        
        # Create manager with mocked dependencies
        self.config = {'data_source': 'alpaca'}
        self.manager = RealTimeDataManager(self.symbols, self.config)
        
        # Set up mock return values
        self.mock_alpaca.get_latest_bars.return_value = pd.DataFrame()
    
    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
    
    def test_initialization(self):
        """Test proper initialization of the real-time data manager"""
        self.assertEqual(self.manager.symbols, self.symbols)
        self.assertEqual(self.manager.data_source, self.mock_alpaca)
        self.assertFalse(self.manager.is_running)
    
    def test_create_data_source(self):
        """Test creating different data sources"""
        # Test Alpaca data source
        source = self.manager._create_data_source('alpaca', self.symbols, {})
        self.assertEqual(source, self.mock_alpaca)
        self.mock_alpaca_class.assert_called_once()
        
        # Test IB data source
        source = self.manager._create_data_source('ib', self.symbols, {})
        self.assertEqual(source, self.mock_ib)
        self.mock_ib_class.assert_called_once()
        
        # Test unsupported data source
        with self.assertRaises(ValueError):
            self.manager._create_data_source('unsupported', self.symbols, {})
    
    def test_on_data_received(self):
        """Test handling incoming market data"""
        # Create a spy for the _update_market_regime method
        self.manager._update_market_regime = MagicMock()
        
        # Test with non-symbol data
        self.manager._on_data_received({})
        self.manager._update_market_regime.assert_not_called()
        
        # Test with symbol data but not a monitored symbol
        data = {'symbol': 'AAPL', 'price': 150.0, 'timestamp': datetime.now()}
        self.manager._on_data_received(data)
        self.manager._update_market_regime.assert_not_called()
        
        # Test with monitored symbol data
        data = {'symbol': 'SPY', 'price': 400.0, 'timestamp': datetime.now().replace(second=0)}
        self.manager._on_data_received(data)
        self.manager._update_market_regime.assert_called_once()
    
    def test_start_stop(self):
        """Test starting and stopping the data manager"""
        # Test start
        self.manager.start()
        self.mock_alpaca.connect.assert_called_once()
        self.mock_alpaca.start_streaming.assert_called_once()
        self.assertTrue(self.manager.is_running)
        
        # Test stop
        self.manager.stop()
        self.mock_alpaca.stop_streaming.assert_called_once()
        self.mock_alpaca.disconnect.assert_called_once()
        self.assertFalse(self.manager.is_running)
    
    def test_get_latest_bars(self):
        """Test retrieving latest bars from the data processor"""
        symbol = 'SPY'
        timeframe = '1min'
        n_bars = 10
        
        # Mock the data processor's get_latest_bars method
        expected_df = pd.DataFrame({'close': [400.0]})
        self.manager.data_processor.get_latest_bars = MagicMock(return_value=expected_df)
        
        # Call the method
        result = self.manager.get_latest_bars(symbol, timeframe, n_bars)
        
        # Verify the result
        self.assertEqual(result, expected_df)
        self.manager.data_processor.get_latest_bars.assert_called_once_with(symbol, timeframe, n_bars)

if __name__ == '__main__':
    unittest.main() 