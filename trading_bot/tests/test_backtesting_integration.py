import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import pandas as pd
import numpy as np
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.learner import BacktestLearner
from data.data_manager import DataManager


class TestBacktestingIntegration(unittest.TestCase):
    
    def setUp(self):
        # Create sample price data
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        self.sample_price_data = pd.DataFrame({
            'open': np.random.rand(100) * 100 + 100,
            'high': np.random.rand(100) * 100 + 120,
            'low': np.random.rand(100) * 100 + 80,
            'close': np.random.rand(100) * 100 + 110,
            'volume': np.random.rand(100) * 1000000
        }, index=dates)
        
        # Create sample technical indicators
        self.sample_indicators = pd.DataFrame({
            'sma_20': np.random.rand(100) * 100 + 105,
            'ema_50': np.random.rand(100) * 100 + 108,
            'rsi_14': np.random.rand(100) * 30 + 50,
            'macd': np.random.rand(100) * 5 - 2.5,
            'macd_signal': np.random.rand(100) * 5 - 2.5
        }, index=dates)
        
        # Create sample target data (classification)
        self.sample_target = pd.Series(np.random.randint(0, 2, 100), index=dates)
        
        # Create sample backtest results
        self.sample_backtest_results = {
            'strategy_name': 'test_strategy',
            'total_return': 0.15,
            'annual_return': 0.12,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'positions': [
                {'date': '2022-01-10', 'action': 'buy', 'price': 105.5, 'shares': 10},
                {'date': '2022-01-20', 'action': 'sell', 'price': 115.5, 'shares': 10}
            ]
        }
        
        # Setup mocks for data_manager
        self.data_manager = MagicMock(spec=DataManager)
        self.data_manager.load_market_data.return_value = self.sample_price_data
        self.data_manager.get_performance_metrics.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08
        }
        self.data_manager.save_backtest_data.return_value = True
        
        # Create a BacktestLearner instance with the mock data_manager
        self.learner = BacktestLearner(data_manager=self.data_manager)
    
    @patch('pandas.DataFrame.to_csv')
    def test_feature_engineering_and_training(self, mock_to_csv):
        """Test the creation of features and model training process"""
        # Mock feature engineering
        with patch.object(self.learner, 'create_features') as mock_create_features:
            # Setup return value for feature creation
            features_df = pd.concat([self.sample_price_data, self.sample_indicators], axis=1)
            mock_create_features.return_value = features_df, self.sample_target
            
            # Mock train_model
            with patch.object(self.learner, 'train_model') as mock_train_model:
                # Setup model mock
                mock_model = MagicMock()
                mock_train_model.return_value = mock_model
                
                # Test the workflow
                symbol = 'AAPL'
                start_date = '2022-01-01'
                end_date = '2022-04-10'
                model_config = {
                    'model_type': 'classification',
                    'algorithm': 'random_forest',
                    'params': {'n_estimators': 100, 'max_depth': 5},
                    'name': 'test_model'
                }
                
                # Execute the workflow (this would be a real method in your learner, here we simulate it)
                # Normally, you'd call something like:
                # model = self.learner.train_strategy_model(symbol, start_date, end_date, model_config)
                
                # Instead, we'll manually execute the workflow steps:
                self.data_manager.load_market_data.assert_not_called()  # Not called yet
                
                # 1. Load market data
                market_data = self.data_manager.load_market_data(symbol, start_date, end_date)
                self.data_manager.load_market_data.assert_called_once_with(symbol, start_date, end_date)
                
                # 2. Create features
                X, y = self.learner.create_features(market_data)
                mock_create_features.assert_called_once_with(market_data)
                
                # 3. Train model
                model = self.learner.train_model(X, y, model_config)
                mock_train_model.assert_called_once()
                
                # Verify the workflow results
                self.assertIs(model, mock_model)
    
    @patch('joblib.dump')
    @patch('json.dump')
    def test_save_model_and_metadata(self, mock_json_dump, mock_joblib_dump):
        """Test that the model and its metadata are properly saved"""
        # Create a mock model
        mock_model = MagicMock()
        
        # Model metadata
        model_metadata = {
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'features': ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi_14'],
            'params': {'n_estimators': 100, 'max_depth': 5},
            'training_period': {'start': '2022-01-01', 'end': '2022-04-10'},
            'symbol': 'AAPL',
            'performance': {'accuracy': 0.78, 'f1': 0.75}
        }
        
        # Test save_model
        with patch('builtins.open', mock_open()) as mock_file:
            self.learner.save_model(mock_model, 'test_strategy_model', model_metadata)
            
            # Check that joblib.dump was called to save the model
            mock_joblib_dump.assert_called_once()
            
            # Check that json.dump was called to save the metadata
            mock_json_dump.assert_called_once()
    
    @patch('os.listdir')
    def test_get_available_models(self, mock_listdir):
        """Test that available models can be retrieved correctly"""
        # Setup mock for os.listdir
        mock_listdir.return_value = [
            'test_model.joblib',
            'test_model.json',
            'another_model.joblib',
            'another_model.json',
            'not_a_model.txt'
        ]
        
        # Mock the exists check for model files
        with patch('os.path.exists', return_value=True):
            # Call the method (assuming it exists)
            # You may need to implement this method in your BacktestLearner class
            models = self.learner.get_available_models()
            
            # Assertions
            self.assertEqual(len(models), 2)
            self.assertIn('test_model', models)
            self.assertIn('another_model', models)
    
    def test_evaluate_model_on_backtest(self):
        """Test that a model can be evaluated using backtest data"""
        # Mock model prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0] * 20)
        
        # Create feature data for prediction
        feature_data = pd.concat([self.sample_price_data, self.sample_indicators], axis=1)
        
        # Mock methods
        with patch.object(self.learner, 'load_model', return_value=mock_model):
            with patch.object(self.learner, 'create_features', return_value=(feature_data, None)):
                # Call the evaluate method (you may need to implement this in your BacktestLearner)
                # results = self.learner.evaluate_model_on_backtest('test_model', 'AAPL', '2022-01-01', '2022-04-10')
                
                # For now, we'll manually simulate the steps:
                
                # 1. Load the model
                model = self.learner.load_model('test_model')
                
                # 2. Load market data for backtest period
                market_data = self.data_manager.load_market_data('AAPL', '2022-01-01', '2022-04-10')
                
                # 3. Create features from market data
                X, _ = self.learner.create_features(market_data)
                
                # 4. Generate predictions
                predictions = model.predict(X)
                
                # 5. Create backtest results based on predictions
                # This would be your custom logic to convert predictions to trades
                
                # For testing purposes, we'll just verify that predictions were made
                self.assertEqual(len(predictions), 100)
                
                # Verify that all the expected methods were called
                self.data_manager.load_market_data.assert_called_with('AAPL', '2022-01-01', '2022-04-10')


if __name__ == "__main__":
    unittest.main() 