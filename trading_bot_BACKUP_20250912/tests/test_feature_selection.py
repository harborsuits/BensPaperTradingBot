import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.learner import BacktestLearner


class TestFeatureSelection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data with 10 features
        np.random.seed(42)  # For reproducibility
        self.features = pd.DataFrame({
            f'feature{i}': np.random.randn(100) for i in range(1, 11)
        })
        
        # Create binary classification target
        self.classification_target = pd.Series(np.random.randint(0, 2, 100))
        
        # Create continuous regression target
        self.regression_target = pd.Series(np.random.randn(100))
        
        # Create a mock data manager
        self.data_manager = MagicMock()
        
        # Initialize the BacktestLearner
        self.learner = BacktestLearner(data_manager=self.data_manager)
    
    def test_select_k_best_features(self):
        """Test selecting K best features using statistical tests"""
        # Mock SelectKBest
        select_k_best_mock = MagicMock(spec=SelectKBest)
        # Mock the get_support method to return a boolean array indicating selected features
        select_k_best_mock.get_support.return_value = np.array([
            True, False, True, False, False, 
            True, False, False, True, False
        ])
        # Mock the scores_ attribute to return feature scores
        select_k_best_mock.scores_ = np.array([
            10.5, 3.2, 8.7, 2.1, 1.5, 
            9.3, 4.2, 3.9, 7.8, 2.6
        ])
        
        # Mock the SelectKBest constructor
        with patch('sklearn.feature_selection.SelectKBest', return_value=select_k_best_mock):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.classification_target,
                method='select_k_best',
                k=4,
                scoring='f_classif'
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('feature_scores', result, "Result should include feature scores")
        
        # Check that the selected features match the mock support
        expected_features = ['feature1', 'feature3', 'feature6', 'feature9']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the feature scores match the mock scores
        expected_scores = {
            'feature1': 10.5, 'feature2': 3.2, 'feature3': 8.7, 'feature4': 2.1, 'feature5': 1.5,
            'feature6': 9.3, 'feature7': 4.2, 'feature8': 3.9, 'feature9': 7.8, 'feature10': 2.6
        }
        self.assertEqual(result['feature_scores'], expected_scores)
    
    def test_recursive_feature_elimination(self):
        """Test recursive feature elimination (RFE)"""
        # Mock RFE
        rfe_mock = MagicMock(spec=RFE)
        # Mock the get_support method to return a boolean array indicating selected features
        rfe_mock.get_support.return_value = np.array([
            True, False, True, False, True, 
            False, False, True, False, False
        ])
        # Mock the ranking_ attribute to return feature rankings
        rfe_mock.ranking_ = np.array([
            1, 3, 1, 5, 1, 
            2, 6, 1, 4, 7
        ])
        
        # Mock the RFE constructor
        with patch('sklearn.feature_selection.RFE', return_value=rfe_mock):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.regression_target,
                method='rfe',
                estimator_type='regression',
                n_features_to_select=4
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('feature_rankings', result, "Result should include feature rankings")
        
        # Check that the selected features match the mock support
        expected_features = ['feature1', 'feature3', 'feature5', 'feature8']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the feature rankings match the mock rankings
        expected_rankings = {
            'feature1': 1, 'feature2': 3, 'feature3': 1, 'feature4': 5, 'feature5': 1,
            'feature6': 2, 'feature7': 6, 'feature8': 1, 'feature9': 4, 'feature10': 7
        }
        self.assertEqual(result['feature_rankings'], expected_rankings)
    
    def test_model_based_selection(self):
        """Test model-based feature selection using feature importances"""
        # Mock SelectFromModel
        select_from_model_mock = MagicMock(spec=SelectFromModel)
        # Mock the get_support method to return a boolean array indicating selected features
        select_from_model_mock.get_support.return_value = np.array([
            True, False, True, False, False, 
            True, True, False, False, False
        ])
        
        # Mock the estimator with feature_importances_
        estimator_mock = MagicMock()
        estimator_mock.feature_importances_ = np.array([
            0.15, 0.05, 0.12, 0.03, 0.07, 
            0.18, 0.14, 0.09, 0.08, 0.09
        ])
        select_from_model_mock.estimator_ = estimator_mock
        
        # Mock the SelectFromModel constructor
        with patch('sklearn.feature_selection.SelectFromModel', return_value=select_from_model_mock):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.classification_target,
                method='model_based',
                estimator_type='classification',
                threshold='mean'
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('feature_importances', result, "Result should include feature importances")
        
        # Check that the selected features match the mock support
        expected_features = ['feature1', 'feature3', 'feature6', 'feature7']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the feature importances match the mock importances
        expected_importances = {
            'feature1': 0.15, 'feature2': 0.05, 'feature3': 0.12, 'feature4': 0.03, 'feature5': 0.07,
            'feature6': 0.18, 'feature7': 0.14, 'feature8': 0.09, 'feature9': 0.08, 'feature10': 0.09
        }
        self.assertEqual(result['feature_importances'], expected_importances)
    
    def test_forward_selection(self):
        """Test sequential forward selection"""
        # Skip this test if forward selection is not implemented
        if not hasattr(self.learner, 'sequential_feature_selection'):
            self.skipTest("Forward selection not implemented")
            return
        
        # Mock the sequential selector
        with patch.object(self.learner, 'sequential_feature_selection', return_value={
            'selected_features': ['feature6', 'feature1', 'feature3', 'feature7'],
            'feature_scores': {
                'feature1': 0.82, 'feature3': 0.78, 'feature6': 0.85, 'feature7': 0.75
            }
        }):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.classification_target,
                method='sequential_forward',
                estimator_type='classification',
                n_features_to_select=4,
                cv=5,
                scoring='accuracy'
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('feature_scores', result, "Result should include feature scores")
        
        # Check that the selected features match the expected features
        expected_features = ['feature1', 'feature3', 'feature6', 'feature7']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
    
    def test_backward_selection(self):
        """Test sequential backward selection"""
        # Skip this test if backward selection is not implemented
        if not hasattr(self.learner, 'sequential_feature_selection'):
            self.skipTest("Backward selection not implemented")
            return
        
        # Mock the sequential selector
        with patch.object(self.learner, 'sequential_feature_selection', return_value={
            'selected_features': ['feature1', 'feature3', 'feature6', 'feature7', 'feature9'],
            'feature_scores': {
                'feature1': 0.82, 'feature3': 0.78, 'feature6': 0.85,
                'feature7': 0.75, 'feature9': 0.73
            }
        }):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.classification_target,
                method='sequential_backward',
                estimator_type='classification',
                n_features_to_select=5,
                cv=5,
                scoring='accuracy'
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('feature_scores', result, "Result should include feature scores")
        
        # Check that the selected features match the expected features
        expected_features = ['feature1', 'feature3', 'feature6', 'feature7', 'feature9']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
    
    def test_correlation_based_selection(self):
        """Test correlation-based feature selection"""
        # Mock the correlation calculation
        correlation_mock = pd.DataFrame({
            'target': [0.8, 0.1, 0.7, 0.2, 0.3, 0.6, 0.75, 0.15, 0.4, 0.05]
        }, index=[f'feature{i}' for i in range(1, 11)])
        
        # Mock the corr method
        with patch.object(pd.DataFrame, 'corr', return_value=correlation_mock):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.regression_target,
                method='correlation',
                threshold=0.5
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('correlations', result, "Result should include correlations")
        
        # Check that the selected features match the expected features (those with correlation > 0.5)
        expected_features = ['feature1', 'feature3', 'feature6', 'feature7']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the correlations match the mock correlations
        expected_correlations = {
            'feature1': 0.8, 'feature2': 0.1, 'feature3': 0.7, 'feature4': 0.2, 'feature5': 0.3,
            'feature6': 0.6, 'feature7': 0.75, 'feature8': 0.15, 'feature9': 0.4, 'feature10': 0.05
        }
        self.assertEqual(result['correlations'], expected_correlations)
    
    def test_mutual_information_selection(self):
        """Test mutual information-based feature selection"""
        # Mock mutual information calculation
        mutual_info_mock = np.array([0.4, 0.1, 0.35, 0.15, 0.2, 0.3, 0.25, 0.05, 0.12, 0.08])
        
        # Mock the mutual_info_classif function
        with patch('sklearn.feature_selection.mutual_info_classif', return_value=mutual_info_mock):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=self.classification_target,
                method='mutual_information',
                k=3
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('mutual_info_scores', result, "Result should include mutual information scores")
        
        # Check that the selected features match the expected features (top 3 by mutual info)
        expected_features = ['feature1', 'feature3', 'feature6']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the mutual info scores match the mock scores
        expected_scores = {
            'feature1': 0.4, 'feature2': 0.1, 'feature3': 0.35, 'feature4': 0.15, 'feature5': 0.2,
            'feature6': 0.3, 'feature7': 0.25, 'feature8': 0.05, 'feature9': 0.12, 'feature10': 0.08
        }
        self.assertEqual(result['mutual_info_scores'], expected_scores)
    
    def test_variance_threshold_selection(self):
        """Test variance threshold feature selection"""
        # Mock variance calculation
        variances = np.array([0.8, 0.9, 0.3, 0.2, 0.5, 0.6, 0.1, 0.4, 0.95, 0.85])
        
        # Mock the var method
        with patch.object(pd.DataFrame, 'var', return_value=pd.Series(
            variances, index=[f'feature{i}' for i in range(1, 11)]
        )):
            # Call the feature selection method
            result = self.learner.feature_selection(
                X=self.features,
                y=None,  # Not needed for variance threshold
                method='variance_threshold',
                threshold=0.5
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('variances', result, "Result should include variances")
        
        # Check that the selected features match the expected features (those with variance > 0.5)
        expected_features = ['feature1', 'feature2', 'feature6', 'feature9', 'feature10']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the variances match the mock variances
        expected_variances = {
            'feature1': 0.8, 'feature2': 0.9, 'feature3': 0.3, 'feature4': 0.2, 'feature5': 0.5,
            'feature6': 0.6, 'feature7': 0.1, 'feature8': 0.4, 'feature9': 0.95, 'feature10': 0.85
        }
        self.assertEqual(result['variances'], expected_variances)
    
    def test_feature_selection_with_multiple_methods(self):
        """Test combining multiple feature selection methods"""
        # Skip this test if combined feature selection is not implemented
        if not hasattr(self.learner, 'combine_feature_selection_methods'):
            self.skipTest("Combined feature selection not implemented")
            return
        
        # Mock individual feature selection methods
        method1_result = {
            'selected_features': ['feature1', 'feature3', 'feature6', 'feature7'],
            'feature_scores': {'feature1': 0.8, 'feature3': 0.7, 'feature6': 0.6, 'feature7': 0.75}
        }
        
        method2_result = {
            'selected_features': ['feature1', 'feature2', 'feature3', 'feature9'],
            'feature_scores': {'feature1': 0.85, 'feature2': 0.6, 'feature3': 0.7, 'feature9': 0.65}
        }
        
        # Mock the individual feature selection methods
        with patch.object(self.learner, 'feature_selection', side_effect=[method1_result, method2_result]):
            # Call the combined feature selection method
            result = self.learner.combine_feature_selection_methods(
                X=self.features,
                y=self.classification_target,
                methods=['select_k_best', 'model_based'],
                combine_method='intersection'
            )
        
        # Check that the result has the expected format
        self.assertIn('selected_features', result, "Result should include selected features")
        self.assertIn('method_results', result, "Result should include individual method results")
        
        # Check that the selected features match the expected features (intersection of both methods)
        expected_features = ['feature1', 'feature3']
        self.assertEqual(sorted(result['selected_features']), sorted(expected_features))
        
        # Check that the method results match the individual method results
        expected_method_results = {
            'select_k_best': method1_result,
            'model_based': method2_result
        }
        self.assertEqual(result['method_results'], expected_method_results)


if __name__ == "__main__":
    unittest.main() 