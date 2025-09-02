import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.learner import BacktestLearner


class TestBacktestLearner(unittest.TestCase):
    
    def setUp(self):
        # Create mock data manager
        self.mock_data_manager = MagicMock()
        
        # Create sample data
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        
        self.y_train_classification = pd.Series(np.random.randint(0, 2, 100))
        self.y_train_regression = pd.Series(np.random.rand(100) * 10)
        
        # Create a test BacktestLearner instance
        self.learner = BacktestLearner(data_manager=self.mock_data_manager)
    
    def test_initialization(self):
        # Test learner initialization
        self.assertIsInstance(self.learner, BacktestLearner)
        self.assertEqual(self.learner.data_manager, self.mock_data_manager)
    
    @patch('sklearn.ensemble.RandomForestClassifier.fit')
    def test_train_classification_model(self, mock_fit):
        # Test training a classification model
        model_config = {
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'params': {'n_estimators': 100, 'max_depth': 5},
            'name': 'test_model'
        }
        
        model = self.learner.train_model(self.X_train, self.y_train_classification, model_config)
        
        # Assertions
        self.assertIsInstance(model, RandomForestClassifier)
        mock_fit.assert_called_once()
    
    @patch('joblib.dump')
    def test_save_model(self, mock_dump):
        # Create a model to save
        model = RandomForestClassifier()
        model_metadata = {
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'features': ['feature1', 'feature2', 'feature3'],
            'params': {'n_estimators': 100}
        }
        
        # Test saving the model
        self.learner.save_model(model, 'test_model', model_metadata)
        
        # Assertions
        mock_dump.assert_called_once()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_model(self, mock_exists, mock_load):
        # Setup mocks
        mock_exists.return_value = True
        mock_model = RandomForestClassifier()
        mock_load.return_value = mock_model
        
        # Test loading the model
        model = self.learner.load_model('test_model')
        
        # Assertions
        self.assertEqual(model, mock_model)
        mock_load.assert_called_once()
    
    @patch('sklearn.model_selection.cross_val_score')
    def test_cross_validate_model(self, mock_cv_score):
        # Setup mock
        mock_cv_score.return_value = np.array([0.85, 0.82, 0.88, 0.86, 0.84])
        
        # Create model config
        model_config = {
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'params': {'n_estimators': 100}
        }
        
        # Test cross-validation
        cv_scores = self.learner.cross_validate_model(self.X_train, self.y_train_classification, model_config)
        
        # Assertions
        self.assertEqual(len(cv_scores), 5)
        self.assertAlmostEqual(np.mean(cv_scores), 0.85, places=2)
        mock_cv_score.assert_called_once()
    
    def test_evaluate_classification_model(self):
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]
        ])
        
        # Test data
        X_test = self.X_train.iloc[:5]
        y_test = pd.Series([0, 1, 0, 0, 1])
        
        # Test evaluating a classification model
        metrics = self.learner.evaluate_classification_model(mock_model, X_test, y_test)
        
        # Assertions
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
    @patch('sklearn.model_selection.GridSearchCV')
    def test_optimize_hyperparameters(self, mock_grid_search):
        # Setup mock
        mock_grid_instance = MagicMock()
        mock_grid_instance.best_params_ = {'n_estimators': 200, 'max_depth': 10}
        mock_grid_instance.best_score_ = 0.88
        mock_grid_search.return_value = mock_grid_instance
        
        # Create model config with param grid
        model_config = {
            'model_type': 'classification',
            'algorithm': 'random_forest',
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10]
            }
        }
        
        # Test hyperparameter optimization
        best_params, best_score = self.learner.optimize_hyperparameters(
            self.X_train, self.y_train_classification, model_config
        )
        
        # Assertions
        self.assertEqual(best_params['n_estimators'], 200)
        self.assertEqual(best_params['max_depth'], 10)
        self.assertEqual(best_score, 0.88)
        mock_grid_search.assert_called_once()
    
    @patch('sklearn.feature_selection.SelectKBest')
    def test_feature_selection(self, mock_select_k_best):
        # Setup mock
        mock_selector = MagicMock()
        mock_selector.get_support.return_value = np.array([True, False, True])
        mock_select_k_best.return_value = mock_selector
        
        # Test feature selection
        self.X_train.columns = ['feature1', 'feature2', 'feature3']
        selected_features = self.learner.feature_selection(
            self.X_train, self.y_train_classification, method='k_best', k=2
        )
        
        # Assertions
        self.assertEqual(len(selected_features), 2)
        self.assertIn('feature1', selected_features)
        self.assertIn('feature3', selected_features)


if __name__ == "__main__":
    unittest.main() 