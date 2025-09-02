import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import pandas as pd
import numpy as np
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_manager import DataManager


class TestDataManager(unittest.TestCase):
    
    def setUp(self):
        # Create a test DataManager instance
        self.data_manager = DataManager()
        
        # Create sample data
        self.sample_price_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000, 10000, 10)
        })
        
        self.sample_backtest_result = {
            'strategy_name': 'TestStrategy',
            'start_date': '2023-01-01',
            'end_date': '2023-01-10',
            'initial_capital': 10000,
            'final_capital': 12000,
            'trades': [
                {'date': '2023-01-02', 'type': 'buy', 'price': 95.5, 'quantity': 10},
                {'date': '2023-01-05', 'type': 'sell', 'price': 105.5, 'quantity': 10}
            ],
            'metrics': {
                'total_return': 0.2,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.05
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('json.load')
    def test_load_backtest_data(self, mock_json_load, mock_exists, mock_file):
        # Setup mocks
        mock_exists.return_value = True
        mock_json_load.return_value = self.sample_backtest_result
        
        # Test loading backtest data
        result = self.data_manager.load_backtest_data('TestStrategy')
        
        # Assertions
        self.assertEqual(result['strategy_name'], 'TestStrategy')
        self.assertEqual(result['metrics']['total_return'], 0.2)
        self.assertEqual(len(result['trades']), 2)
    
    @patch('pandas.read_csv')
    def test_load_market_data(self, mock_read_csv):
        # Setup mock
        mock_read_csv.return_value = self.sample_price_data
        
        # Test loading market data
        data = self.data_manager.load_market_data('AAPL', '2023-01-01', '2023-01-10')
        
        # Assertions
        self.assertIsInstance(data, pd.DataFrame)
        mock_read_csv.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_backtest_data(self, mock_file):
        # Test saving backtest data
        self.data_manager.save_backtest_data('TestStrategy', self.sample_backtest_result)
        
        # Assertions
        mock_file.assert_called_once()
        
    def test_get_performance_metrics(self):
        # Test getting performance metrics
        with patch.object(self.data_manager, 'load_backtest_data') as mock_load:
            mock_load.return_value = self.sample_backtest_result
            
            metrics = self.data_manager.get_performance_metrics('TestStrategy')
            
            # Assertions
            self.assertEqual(metrics['total_return'], 0.2)
            self.assertEqual(metrics['sharpe_ratio'], 1.5)
            
    def test_get_available_strategies(self):
        # Test getting available strategies
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ['TestStrategy.json', 'AnotherStrategy.json']
            
            strategies = self.data_manager.get_available_strategies()
            
            # Assertions
            self.assertEqual(len(strategies), 2)
            self.assertIn('TestStrategy', strategies)
            self.assertIn('AnotherStrategy', strategies)


if __name__ == "__main__":
    unittest.main() 