import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.learner import BacktestLearner
from data.data_manager import DataManager


class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate sample OHLCV data
        self.dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'open': np.random.rand(100) * 100 + 100,
            'high': np.random.rand(100) * 100 + 120,
            'low': np.random.rand(100) * 100 + 80,
            'close': np.random.rand(100) * 100 + 110,
            'volume': np.random.rand(100) * 1000000
        }, index=self.dates)
        
        # Create a DataManager mock
        self.data_manager = unittest.mock.MagicMock(spec=DataManager)
        self.data_manager.load_market_data.return_value = self.sample_data
        
        # Initialize the BacktestLearner
        self.learner = BacktestLearner(data_manager=self.data_manager)
    
    def test_technical_indicator_calculation(self):
        """Test that technical indicators are correctly calculated"""
        # Call the create_features method or appropriate method in your BacktestLearner
        features, _ = self.learner.create_features(self.sample_data)
        
        # Check that the expected technical indicators exist in the features
        expected_indicators = [
            'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi_14', 'bollinger_upper', 'bollinger_lower', 'bollinger_pct_b'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, features.columns, f"Missing indicator: {indicator}")
        
        # Verify that indicators have appropriate values
        # Example: RSI should be between 0 and 100
        self.assertTrue((features['rsi_14'] >= 0).all() and (features['rsi_14'] <= 100).all(),
                        "RSI values should be between 0 and 100")
        
        # Bollinger Bands: Upper should be > Lower
        self.assertTrue((features['bollinger_upper'] > features['bollinger_lower']).all(),
                       "Bollinger upper band should be greater than lower band")
    
    def test_target_variable_creation(self):
        """Test that target variables are correctly created for ML models"""
        # Call the method that creates target variables
        _, target = self.learner.create_features(self.sample_data, 
                                                target_type='classification',
                                                lookahead_period=5, 
                                                threshold=0.01)
        
        # Check that target has the correct length
        # For classification with lookahead, we typically lose some rows at the end
        expected_length = len(self.sample_data) - 5
        self.assertEqual(len(target), expected_length,
                         f"Target length should be {expected_length}, got {len(target)}")
        
        # Check that the target values are binary (0 or 1) for classification
        unique_values = target.unique()
        self.assertTrue(set(unique_values).issubset({0, 1}),
                       f"Classification target should contain only 0 and 1, got {unique_values}")
    
    def test_data_normalization(self):
        """Test that feature data is properly normalized"""
        # Call the feature creation with normalization
        features, _ = self.learner.create_features(self.sample_data, normalize=True)
        
        # Check that price columns are normalized
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in features.columns:
                # Normalized values should typically be between -3 and 3 for most data
                self.assertTrue(abs(features[col].mean()) < 3,
                              f"Normalized {col} mean should be close to 0")
                self.assertTrue(features[col].std() < 3,
                              f"Normalized {col} std should be reasonable")
    
    def test_feature_lag_creation(self):
        """Test that lagged features are correctly created"""
        # Call the feature creation with lag parameters
        lags = [1, 2, 3]
        features, _ = self.learner.create_features(self.sample_data, lags=lags)
        
        # Check that lagged columns exist
        for col in ['close', 'volume']:  # Example columns that might be lagged
            for lag in lags:
                lagged_col = f"{col}_lag_{lag}"
                self.assertIn(lagged_col, features.columns,
                             f"Missing lagged column: {lagged_col}")
                
                # Verify lag relationship (first non-NaN values should match)
                # Skip NaN values at the beginning due to lag
                non_nan_idx = lag
                original_value = self.sample_data[col].iloc[non_nan_idx - lag]
                lagged_value = features[lagged_col].iloc[non_nan_idx]
                self.assertAlmostEqual(original_value, lagged_value,
                                     msg=f"Lag relationship incorrect for {lagged_col}")
    
    def test_returns_calculation(self):
        """Test that return calculations are correct"""
        # Call the feature creation with returns calculation
        features, _ = self.learner.create_features(self.sample_data, calculate_returns=True)
        
        # Check that returns columns exist
        for period in [1, 5]:  # Example return periods
            returns_col = f"returns_{period}d"
            self.assertIn(returns_col, features.columns,
                         f"Missing returns column: {returns_col}")
            
            # Calculate expected returns manually for verification
            expected_returns = self.sample_data['close'].pct_change(period)
            
            # Compare with calculated returns (allowing for small floating point differences)
            pd.testing.assert_series_equal(features[returns_col], expected_returns,
                                         check_names=False, check_dtype=False,
                                         check_exact=False, rtol=1e-5)
    
    def test_missing_value_handling(self):
        """Test that missing values are properly handled"""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.iloc[10:15, 0:2] = np.nan  # Set some values to NaN
        
        # Call feature creation on data with missing values
        features, target = self.learner.create_features(data_with_missing)
        
        # Check that there are no NaN values in the features
        self.assertFalse(features.isnull().any().any(),
                        "Features should not contain NaN values after processing")
    
    def test_volatility_features(self):
        """Test that volatility features are correctly calculated"""
        # Call the feature creation with volatility calculation
        features, _ = self.learner.create_features(self.sample_data, calculate_volatility=True)
        
        # Check that volatility columns exist
        volatility_cols = ['volatility_5d', 'volatility_20d']
        for col in volatility_cols:
            self.assertIn(col, features.columns, f"Missing volatility column: {col}")
            
            # Volatility should be non-negative
            self.assertTrue((features[col] >= 0).all(),
                          f"Volatility values should be non-negative")
            
            # Higher period volatility should be generally smoother
            if 'volatility_5d' in features.columns and 'volatility_20d' in features.columns:
                self.assertTrue(features['volatility_5d'].std() >= features['volatility_20d'].std(),
                              "Short-term volatility should generally be more variable than long-term")


if __name__ == "__main__":
    unittest.main() 