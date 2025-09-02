import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.learner import BacktestLearner


class TestHyperparameterOptimization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        self.features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        # Create binary classification target
        self.classification_target = pd.Series(np.random.randint(0, 2, 100))
        
        # Create continuous regression target
        self.regression_target = pd.Series(np.random.randn(100))
        
        # Create a mock data manager
        self.data_manager = MagicMock()
        
        # Initialize the BacktestLearner
        self.learner = BacktestLearner(data_manager=self.data_manager)
    
    def test_grid_search_optimization(self):
        """Test that grid search hyperparameter optimization is performed correctly"""
        # Define hyperparameters for grid search
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10]
        }
        
        # Mock GridSearchCV
        grid_search_mock = MagicMock(spec=GridSearchCV)
        grid_search_mock.best_params_ = {'n_estimators': 50, 'max_depth': 5}
        grid_search_mock.best_score_ = 0.85
        grid_search_mock.cv_results_ = {
            'mean_test_score': np.array([0.75, 0.78, 0.82, 0.85, 0.80, 0.83]),
            'params': [
                {'n_estimators': 10, 'max_depth': None},
                {'n_estimators': 10, 'max_depth': 5},
                {'n_estimators': 10, 'max_depth': 10},
                {'n_estimators': 50, 'max_depth': 5},
                {'n_estimators': 100, 'max_depth': None},
                {'n_estimators': 100, 'max_depth': 10}
            ]
        }
        
        # Mock the grid search creation
        with patch('sklearn.model_selection.GridSearchCV', return_value=grid_search_mock):
            # Call the hyperparameter optimization method with grid search
            result = self.learner.optimize_hyperparameters(
                X=self.features,
                y=self.classification_target,
                model_type='classification',
                param_grid=param_grid,
                search_method='grid',
                cv=5,
                scoring='accuracy'
            )
        
        # Check that the result has the expected format
        self.assertIn('best_params', result, "Result should include best parameters")
        self.assertIn('best_score', result, "Result should include best score")
        self.assertIn('all_results', result, "Result should include all results")
        
        # Check that the best parameters and score match the mock values
        self.assertEqual(result['best_params'], {'n_estimators': 50, 'max_depth': 5})
        self.assertEqual(result['best_score'], 0.85)
    
    def test_random_search_optimization(self):
        """Test that random search hyperparameter optimization is performed correctly"""
        # Define hyperparameters for random search
        param_distributions = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 3, 5, 10, 15]
        }
        
        # Mock RandomizedSearchCV
        random_search_mock = MagicMock(spec=RandomizedSearchCV)
        random_search_mock.best_params_ = {'n_estimators': 100, 'max_depth': 10}
        random_search_mock.best_score_ = 0.88
        random_search_mock.cv_results_ = {
            'mean_test_score': np.array([0.78, 0.82, 0.88, 0.85]),
            'params': [
                {'n_estimators': 50, 'max_depth': 5},
                {'n_estimators': 100, 'max_depth': 3},
                {'n_estimators': 100, 'max_depth': 10},
                {'n_estimators': 200, 'max_depth': 15}
            ]
        }
        
        # Mock the random search creation
        with patch('sklearn.model_selection.RandomizedSearchCV', return_value=random_search_mock):
            # Call the hyperparameter optimization method with random search
            result = self.learner.optimize_hyperparameters(
                X=self.features,
                y=self.classification_target,
                model_type='classification',
                param_grid=param_distributions,
                search_method='random',
                cv=5,
                n_iter=10,
                scoring='accuracy'
            )
        
        # Check that the result has the expected format
        self.assertIn('best_params', result, "Result should include best parameters")
        self.assertIn('best_score', result, "Result should include best score")
        self.assertIn('all_results', result, "Result should include all results")
        
        # Check that the best parameters and score match the mock values
        self.assertEqual(result['best_params'], {'n_estimators': 100, 'max_depth': 10})
        self.assertEqual(result['best_score'], 0.88)
    
    def test_regression_model_optimization(self):
        """Test hyperparameter optimization for regression models"""
        # Define hyperparameters for grid search
        param_grid = {
            'alpha': [0.1, 0.5, 1.0],
            'fit_intercept': [True, False]
        }
        
        # Mock GridSearchCV for regression
        grid_search_mock = MagicMock(spec=GridSearchCV)
        grid_search_mock.best_params_ = {'alpha': 0.5, 'fit_intercept': True}
        grid_search_mock.best_score_ = -0.25  # Negative MSE score
        grid_search_mock.cv_results_ = {
            'mean_test_score': np.array([-0.42, -0.38, -0.32, -0.25, -0.30, -0.35]),
            'params': [
                {'alpha': 0.1, 'fit_intercept': True},
                {'alpha': 0.1, 'fit_intercept': False},
                {'alpha': 0.5, 'fit_intercept': False},
                {'alpha': 0.5, 'fit_intercept': True},
                {'alpha': 1.0, 'fit_intercept': True},
                {'alpha': 1.0, 'fit_intercept': False}
            ]
        }
        
        # Mock the grid search creation
        with patch('sklearn.model_selection.GridSearchCV', return_value=grid_search_mock):
            # Call the hyperparameter optimization method for regression
            result = self.learner.optimize_hyperparameters(
                X=self.features,
                y=self.regression_target,
                model_type='regression',
                param_grid=param_grid,
                search_method='grid',
                cv=5,
                scoring='neg_mean_squared_error'
            )
        
        # Check that the best parameters and score match the mock values
        self.assertEqual(result['best_params'], {'alpha': 0.5, 'fit_intercept': True})
        self.assertEqual(result['best_score'], -0.25)
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization for hyperparameters (if implemented)"""
        # Skip this test if Bayesian optimization is not implemented
        if not hasattr(self.learner, 'bayesian_optimize') and not hasattr(self.learner, 'optimize_hyperparameters_bayesian'):
            self.skipTest("Bayesian optimization not implemented")
            return
        
        # Define hyperparameters for Bayesian optimization
        param_space = {
            'n_estimators': (10, 200),
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3)
        }
        
        # Mock the Bayesian optimization function
        with patch.object(self.learner, 'bayesian_optimize', return_value={
            'best_params': {'n_estimators': 120, 'max_depth': 8, 'learning_rate': 0.15},
            'best_score': 0.92,
            'all_results': {
                'scores': [0.85, 0.87, 0.90, 0.92],
                'params': [
                    {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                    {'n_estimators': 80, 'max_depth': 7, 'learning_rate': 0.2},
                    {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1},
                    {'n_estimators': 120, 'max_depth': 8, 'learning_rate': 0.15}
                ]
            }
        }):
            # Call the hyperparameter optimization method with Bayesian search
            result = self.learner.optimize_hyperparameters(
                X=self.features,
                y=self.classification_target,
                model_type='classification',
                param_grid=param_space,
                search_method='bayesian',
                cv=5,
                n_iter=20,
                scoring='accuracy'
            )
        
        # Check that the result has the expected format
        self.assertIn('best_params', result, "Result should include best parameters")
        self.assertIn('best_score', result, "Result should include best score")
        
        # Check that the best parameters and score match the mock values
        self.assertEqual(result['best_params']['n_estimators'], 120)
        self.assertEqual(result['best_params']['max_depth'], 8)
        self.assertEqual(result['best_params']['learning_rate'], 0.15)
        self.assertEqual(result['best_score'], 0.92)
    
    def test_validation_curve_generation(self):
        """Test generation of validation curves for hyperparameter tuning"""
        # Mock validation curve function
        train_scores = np.array([0.82, 0.85, 0.88, 0.91, 0.93])
        test_scores = np.array([0.78, 0.82, 0.86, 0.85, 0.83])
        param_values = np.array([1, 10, 50, 100, 200])
        
        with patch('sklearn.model_selection.validation_curve', return_value=(train_scores, test_scores)):
            # Call the validation curve method
            curves = self.learner.generate_validation_curve(
                X=self.features,
                y=self.classification_target,
                model_type='classification',
                param_name='n_estimators',
                param_range=param_values,
                cv=5,
                scoring='accuracy'
            )
        
        # Check that the curves data has the expected format
        self.assertIn('train_scores', curves, "Curves should include training scores")
        self.assertIn('test_scores', curves, "Curves should include test scores")
        self.assertIn('param_values', curves, "Curves should include parameter values")
        
        # Check that the arrays match our mock values
        np.testing.assert_array_equal(curves['train_scores'], train_scores)
        np.testing.assert_array_equal(curves['test_scores'], test_scores)
        np.testing.assert_array_equal(curves['param_values'], param_values)
    
    def test_learning_curve_generation(self):
        """Test generation of learning curves for model evaluation"""
        # Mock learning curve function
        train_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * 80  # 80% of 100 samples
        train_scores = np.array([0.95, 0.93, 0.92, 0.90, 0.89])
        test_scores = np.array([0.75, 0.78, 0.82, 0.84, 0.85])
        
        with patch('sklearn.model_selection.learning_curve', return_value=(train_sizes, train_scores, test_scores)):
            # Call the learning curve method
            curves = self.learner.generate_learning_curve(
                X=self.features,
                y=self.classification_target,
                model_type='classification',
                cv=5,
                scoring='accuracy',
                train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
        
        # Check that the curves data has the expected format
        self.assertIn('train_sizes', curves, "Curves should include training sizes")
        self.assertIn('train_scores', curves, "Curves should include training scores")
        self.assertIn('test_scores', curves, "Curves should include test scores")
        
        # Check that the arrays match our mock values
        np.testing.assert_array_equal(curves['train_sizes'], train_sizes)
        np.testing.assert_array_equal(curves['train_scores'], train_scores)
        np.testing.assert_array_equal(curves['test_scores'], test_scores)
    
    def test_feature_selection_optimization(self):
        """Test feature selection during hyperparameter optimization"""
        # Define hyperparameters including feature selection
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'feature_selection__k': [1, 2, 3]  # Select top k features
        }
        
        # Mock feature selector
        feature_selector = MagicMock()
        feature_selector.get_support.return_value = np.array([True, False, True])
        
        # Mock pipeline with feature selection
        pipeline_mock = MagicMock()
        pipeline_mock.named_steps = {
            'feature_selection': feature_selector
        }
        
        # Mock GridSearchCV
        grid_search_mock = MagicMock(spec=GridSearchCV)
        grid_search_mock.best_params_ = {'n_estimators': 100, 'max_depth': 5, 'feature_selection__k': 2}
        grid_search_mock.best_score_ = 0.88
        grid_search_mock.best_estimator_ = pipeline_mock
        
        # Mock the pipeline and grid search creation
        with patch('sklearn.pipeline.Pipeline'):
            with patch('sklearn.model_selection.GridSearchCV', return_value=grid_search_mock):
                # Call the hyperparameter optimization method with feature selection
                result = self.learner.optimize_hyperparameters_with_feature_selection(
                    X=self.features,
                    y=self.classification_target,
                    model_type='classification',
                    param_grid=param_grid,
                    search_method='grid',
                    cv=5,
                    scoring='accuracy'
                )
        
        # Check that the result includes selected features
        self.assertIn('best_params', result, "Result should include best parameters")
        self.assertIn('best_score', result, "Result should include best score")
        self.assertIn('selected_features', result, "Result should include selected features")
        
        # Check that the selected features match the mock support
        self.assertEqual(result['selected_features'], ['feature1', 'feature3'])


if __name__ == "__main__":
    unittest.main() 