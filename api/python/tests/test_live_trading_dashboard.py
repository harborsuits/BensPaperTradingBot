#!/usr/bin/env python3
"""
Unit tests for the live trading dashboard.

These tests validate the functionality of the dashboard components, data handling,
and visualization aspects of the live trading dashboard.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import queue
from collections import deque

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import with streamlit and plotly mocked
with patch('streamlit.set_page_config'), \
     patch('streamlit.title'), \
     patch('streamlit.markdown'), \
     patch('streamlit.sidebar'), \
     patch('plotly.express.line'), \
     patch('plotly.express.bar'), \
     patch('plotly.express.pie'), \
     patch('plotly.graph_objects.Figure'):
    
    from trading_bot.visualization.live_trading_dashboard import (
        DashboardState,
        initialize_dashboard,
        start_data_streaming,
        stop_data_streaming,
        on_bar_update,
        on_regime_change,
        on_strategy_update,
        update_mock_portfolio,
        simulate_trade,
        create_mock_data_thread,
        display_portfolio_performance,
        display_market_data,
        display_market_regime,
        display_trading_activity,
        create_dashboard
    )

class TestDashboardState(unittest.TestCase):
    """Tests for the DashboardState class"""
    
    def test_initialization(self):
        """Test proper initialization of dashboard state"""
        state = DashboardState()
        
        # Verify basic attributes
        self.assertIsInstance(state.data_queue, queue.Queue)
        self.assertIsInstance(state.portfolio_history, deque)
        self.assertIsInstance(state.trade_history, deque)
        self.assertIsInstance(state.market_regime_history, deque)
        self.assertFalse(state.streaming_active)
        self.assertIsNone(state.data_manager)
        self.assertEqual(state.symbols, [])
        
        # Verify deque maxlen
        self.assertEqual(state.portfolio_history.maxlen, 1000)
        self.assertEqual(state.trade_history.maxlen, 100)
        self.assertEqual(state.market_regime_history.maxlen, 100)

class TestDashboardFunctions(unittest.TestCase):
    """Tests for the dashboard functions"""
    
    def setUp(self):
        # Create a fresh dashboard state for each test
        self.original_state = sys.modules['trading_bot.visualization.live_trading_dashboard'].state
        self.state = DashboardState()
        sys.modules['trading_bot.visualization.live_trading_dashboard'].state = self.state
    
    def tearDown(self):
        # Restore the original state
        sys.modules['trading_bot.visualization.live_trading_dashboard'].state = self.original_state
    
    @patch('streamlit.sidebar')
    @patch('streamlit.button')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    @patch('streamlit.slider')
    def test_initialize_dashboard(self, mock_slider, mock_text_input, mock_selectbox, mock_button, mock_sidebar):
        """Test dashboard initialization with streamlit components"""
        # Mock streamlit component returns
        mock_selectbox.side_effect = ["Alpaca", "1 Day"]
        mock_text_input.return_value = "SPY,QQQ"
        mock_slider.return_value = 5
        mock_button.return_value = False
        
        # Call the function
        symbols, update_frequency, time_range, data_source = initialize_dashboard()
        
        # Verify the results
        self.assertEqual(symbols, ["SPY", "QQQ"])
        self.assertEqual(update_frequency, 5)
        self.assertEqual(time_range, "1 Day")
        self.assertEqual(data_source, "Alpaca")
    
    @patch('trading_bot.visualization.live_trading_dashboard.RealTimeDataManager')
    def test_start_data_streaming_with_alpaca(self, mock_manager):
        """Test starting data streaming with Alpaca"""
        symbols = ["SPY", "QQQ"]
        data_source = "Alpaca"
        
        # Mock the data manager
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        
        # Call the function
        with patch.dict(os.environ, {'ALPACA_API_KEY': 'test-key', 'ALPACA_API_SECRET': 'test-secret'}):
            start_data_streaming(symbols, data_source)
        
        # Verify the data manager was initialized correctly
        self.assertEqual(self.state.symbols, symbols)
        self.assertTrue(self.state.streaming_active)
        self.assertEqual(self.state.data_manager, mock_manager_instance)
        
        # Verify the data manager methods were called
        mock_manager.assert_called_once()
        mock_manager_instance.start.assert_called_once()
        
        # Verify callbacks were registered
        self.assertEqual(mock_manager_instance.on_bar_update, on_bar_update)
        self.assertEqual(mock_manager_instance.on_regime_change, on_regime_change)
        self.assertEqual(mock_manager_instance.on_strategy_update, on_strategy_update)
    
    @patch('trading_bot.visualization.live_trading_dashboard.create_mock_data_thread')
    def test_start_data_streaming_with_mock_data(self, mock_create_thread):
        """Test starting data streaming with mock data"""
        symbols = ["SPY", "QQQ"]
        data_source = "Mock Data"
        
        # Call the function
        start_data_streaming(symbols, data_source)
        
        # Verify mock data generator was started
        mock_create_thread.assert_called_once_with(symbols, self.state.data_queue)
        self.assertTrue(self.state.streaming_active)
        self.assertIsNone(self.state.data_manager)
    
    def test_stop_data_streaming(self):
        """Test stopping data streaming"""
        # Setup mock data manager
        mock_manager = MagicMock()
        self.state.data_manager = mock_manager
        self.state.streaming_active = True
        
        # Call the function
        stop_data_streaming()
        
        # Verify the data manager was stopped
        mock_manager.stop.assert_called_once()
        self.assertFalse(self.state.streaming_active)
        self.assertIsNone(self.state.data_manager)
    
    def test_on_bar_update(self):
        """Test handling bar updates"""
        # Setup initial state
        mock_update_portfolio = MagicMock()
        sys.modules['trading_bot.visualization.live_trading_dashboard'].update_mock_portfolio = mock_update_portfolio
        
        # Create test data
        data = {
            'symbol': 'SPY',
            'price': 400.0,
            'timestamp': datetime.now()
        }
        
        # Call the function
        on_bar_update(data)
        
        # Verify data was added to queue
        self.assertEqual(self.state.data_queue.qsize(), 1)
        queue_item = self.state.data_queue.get()
        self.assertEqual(queue_item['type'], 'bar_update')
        self.assertEqual(queue_item['data'], data)
    
    def test_on_regime_change(self):
        """Test handling regime changes"""
        # Call the function with a test regime
        on_regime_change('bull')
        
        # Verify regime was added to history
        self.assertEqual(len(self.state.market_regime_history), 1)
        self.assertEqual(self.state.market_regime_history[0]['regime'], 'bull')
        
        # Verify data was added to queue
        self.assertEqual(self.state.data_queue.qsize(), 1)
        queue_item = self.state.data_queue.get()
        self.assertEqual(queue_item['type'], 'regime_change')
        self.assertEqual(queue_item['regime'], 'bull')
    
    @patch('trading_bot.visualization.live_trading_dashboard.simulate_trade')
    def test_on_strategy_update(self, mock_simulate_trade):
        """Test handling strategy weight updates"""
        # Call the function with test weights
        weights = {'MA_Trend': 0.5, 'Mean_Reversion': 0.5}
        on_strategy_update(weights)
        
        # Verify weights were added to queue
        self.assertEqual(self.state.data_queue.qsize(), 1)
        queue_item = self.state.data_queue.get()
        self.assertEqual(queue_item['type'], 'strategy_update')
        self.assertEqual(queue_item['weights'], weights)
        
        # Verify trade was simulated
        mock_simulate_trade.assert_called_once_with(weights)
    
    def test_update_mock_portfolio(self):
        """Test updating mock portfolio data"""
        # Setup initial portfolio
        self.state.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': 100000.0,
            'cash': 100000.0,
            'invested': 0.0
        })
        
        # Add a market regime
        self.state.market_regime_history.append({
            'timestamp': datetime.now(),
            'regime': 'bull',
            'duration': 0
        })
        
        # Call the function
        update_mock_portfolio()
        
        # Verify portfolio was updated
        self.assertEqual(len(self.state.portfolio_history), 2)
        self.assertGreater(self.state.portfolio_history[1]['value'], 0)
        self.assertLess(abs(self.state.portfolio_history[1]['value'] - 100000.0), 100)  # Should be close to initial
    
    def test_simulate_trade(self):
        """Test simulating a trade"""
        # Setup symbols
        self.state.symbols = ["SPY", "QQQ"]
        
        # Call the function
        weights = {'MA_Trend': 0.5, 'Mean_Reversion': 0.5}
        simulate_trade(weights)
        
        # Verify trade was added to history
        self.assertEqual(len(self.state.trade_history), 1)
        trade = self.state.trade_history[0]
        self.assertIn(trade['symbol'], self.state.symbols)
        self.assertIn(trade['action'], ['BUY', 'SELL'])
        self.assertGreater(trade['quantity'], 0)
        self.assertGreater(trade['price'], 0)
        self.assertEqual(trade['weights'], weights)
    
    @patch('threading.Thread')
    def test_create_mock_data_thread(self, mock_thread):
        """Test creating a mock data generator thread"""
        # Setup mock thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # Call the function
        symbols = ["SPY", "QQQ"]
        data_queue = queue.Queue()
        create_mock_data_thread(symbols, data_queue)
        
        # Verify thread was created and started
        mock_thread.assert_called_once()
        self.assertTrue(mock_thread.call_args[1]['daemon'])
        mock_thread_instance.start.assert_called_once()

class TestDashboardDisplayFunctions(unittest.TestCase):
    """Tests for the dashboard display functions"""
    
    def setUp(self):
        # Create a fresh dashboard state for each test
        self.original_state = sys.modules['trading_bot.visualization.live_trading_dashboard'].state
        self.state = DashboardState()
        sys.modules['trading_bot.visualization.live_trading_dashboard'].state = self.state
        
        # Add some test data
        for i in range(10):
            self.state.portfolio_history.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'value': 100000.0 + i * 100,
                'cash': 50000.0,
                'invested': 50000.0 + i * 100
            })
            
            if i % 3 == 0:  # Add some regime changes
                self.state.market_regime_history.append({
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'regime': ['bull', 'bear', 'consolidation', 'volatility'][i % 4],
                    'duration': 0
                })
            
            if i % 2 == 0:  # Add some trades
                self.state.trade_history.append({
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'symbol': ['SPY', 'QQQ'][i % 2],
                    'action': ['BUY', 'SELL'][i % 2],
                    'quantity': 10 + i,
                    'price': 400.0 + i,
                    'value': (10 + i) * (400.0 + i),
                    'weights': {'MA_Trend': 0.5, 'Mean_Reversion': 0.5}
                })
        
        # Setup symbols
        self.state.symbols = ["SPY", "QQQ"]
        self.state.streaming_active = True
    
    def tearDown(self):
        # Restore the original state
        sys.modules['trading_bot.visualization.live_trading_dashboard'].state = self.original_state
    
    @patch('streamlit.header')
    @patch('streamlit.info')
    @patch('streamlit.tabs')
    @patch('plotly.express.line')
    @patch('plotly.express.area')
    @patch('plotly.express.pie')
    @patch('streamlit.plotly_chart')
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_display_portfolio_performance(self, mock_columns, mock_metric, mock_plotly_chart, 
                                         mock_pie, mock_area, mock_line, mock_tabs, mock_info, mock_header):
        """Test displaying portfolio performance"""
        # Mock tab context manager
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock columns
        mock_column = MagicMock()
        mock_columns.return_value = [mock_column, mock_column, mock_column, mock_column]
        
        # Call the function
        display_portfolio_performance()
        
        # Verify header was displayed
        mock_header.assert_called_once_with("Portfolio Performance")
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(["Portfolio Value", "Allocation", "Performance Metrics"])
        
        # Verify plots were created
        self.assertGreater(mock_line.call_count, 0)
        self.assertGreater(mock_pie.call_count, 0)
        
        # Verify metrics were displayed
        self.assertGreater(mock_metric.call_count, 0)
    
    @patch('streamlit.header')
    @patch('streamlit.info')
    @patch('streamlit.tabs')
    @patch('plotly.graph_objects.Figure')
    @patch('plotly.express.line')
    @patch('streamlit.plotly_chart')
    def test_display_market_data(self, mock_plotly_chart, mock_line, mock_figure, 
                               mock_tabs, mock_info, mock_header):
        """Test displaying market data"""
        # Mock tab context manager
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        
        # Create mock data manager
        self.state.data_manager = MagicMock()
        self.state.data_manager.get_latest_bars.return_value = pd.DataFrame()
        
        # Call the function
        display_market_data()
        
        # Verify header was displayed
        mock_header.assert_called_once_with("Market Data")
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(self.state.symbols)
    
    @patch('streamlit.header')
    @patch('streamlit.info')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('plotly.express.timeline')
    @patch('plotly.express.bar')
    @patch('streamlit.plotly_chart')
    def test_display_market_regime(self, mock_plotly_chart, mock_bar, mock_timeline, 
                                 mock_dataframe, mock_subheader, mock_columns, mock_markdown, 
                                 mock_info, mock_header):
        """Test displaying market regime information"""
        # Mock column context manager
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Call the function
        display_market_regime()
        
        # Verify header was displayed
        mock_header.assert_called_once_with("Market Regime Analysis")
        
        # Verify current regime was displayed
        mock_markdown.assert_called_once()
        
        # Verify columns were created
        mock_columns.assert_called_once_with(2)
        
        # Verify plots were created
        self.assertGreater(mock_timeline.call_count + mock_bar.call_count, 0)
    
    @patch('streamlit.header')
    @patch('streamlit.info')
    @patch('streamlit.tabs')
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('plotly.express.scatter')
    @patch('plotly.express.bar')
    @patch('plotly.express.pie')
    @patch('plotly.express.line')
    @patch('streamlit.plotly_chart')
    def test_display_trading_activity(self, mock_plotly_chart, mock_line, mock_pie, mock_bar, 
                                    mock_scatter, mock_dataframe, mock_subheader, mock_tabs, 
                                    mock_info, mock_header):
        """Test displaying trading activity"""
        # Mock tab context manager
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]
        
        # Call the function
        display_trading_activity()
        
        # Verify header was displayed
        mock_header.assert_called_once_with("Trading Activity")
        
        # Verify tabs were created
        mock_tabs.assert_called_once_with(["Recent Trades", "Strategy Allocations"])
        
        # Verify plots were created
        self.assertGreater(mock_scatter.call_count + mock_bar.call_count + mock_pie.call_count + mock_line.call_count, 0)
    
    @patch('streamlit.container')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('trading_bot.visualization.live_trading_dashboard.initialize_dashboard')
    @patch('trading_bot.visualization.live_trading_dashboard.display_portfolio_performance')
    @patch('trading_bot.visualization.live_trading_dashboard.display_market_data')
    @patch('trading_bot.visualization.live_trading_dashboard.display_market_regime')
    @patch('trading_bot.visualization.live_trading_dashboard.display_trading_activity')
    @patch('time.sleep')
    @patch('streamlit.experimental_rerun')
    def test_create_dashboard(self, mock_rerun, mock_sleep, mock_display_trading, 
                            mock_display_regime, mock_display_market, mock_display_portfolio, 
                            mock_initialize, mock_metric, mock_columns, mock_container):
        """Test creating the complete dashboard"""
        # Mock dashboard container
        mock_container_ctx = MagicMock()
        mock_container.return_value = mock_container_ctx
        
        # Mock column context manager
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        
        # Mock initialize to return test values
        mock_initialize.return_value = (["SPY", "QQQ"], 5, "1 Day", "Alpaca")
        
        # Set streaming active
        self.state.streaming_active = True
        
        # Call the function
        create_dashboard()
        
        # Verify dashboard was initialized
        mock_initialize.assert_called_once()
        
        # Verify display functions were called
        mock_display_portfolio.assert_called_once()
        mock_display_market.assert_called_once()
        mock_display_regime.assert_called_once()
        mock_display_trading.assert_called_once()
        
        # Verify metrics were displayed
        self.assertGreater(mock_metric.call_count, 0)
        
        # Verify auto-refresh was triggered
        mock_sleep.assert_called_once()
        mock_rerun.assert_called_once()

if __name__ == '__main__':
    unittest.main() 