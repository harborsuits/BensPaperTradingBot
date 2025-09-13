import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import json
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.learner import BacktestLearner


class TestModelEvaluation(unittest.TestCase):
    
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
        
        # Create a sample trained model
        self.model = RandomForestClassifier(n_estimators=10)
        self.model.fit(self.features, self.classification_target)
    
    def test_classification_evaluation_metrics(self):
        """Test that classification metrics are calculated correctly"""
        # Split data for testing
        train_features = self.features.iloc[:70]
        train_target = self.classification_target.iloc[:70]
        test_features = self.features.iloc[70:]
        test_target = self.classification_target.iloc[70:]
        
        # Mock the model's predict and predict_proba methods
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1, 0] * 6)  # 30 predictions
        model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]] * 6)
        
        # Set true labels for comparison
        true_labels = np.array([0, 1, 0, 0, 1] * 6)  # 30 true labels
        
        # Call the evaluation method with classification target
        with patch.object(self.learner, 'get_model', return_value=model):
            metrics = self.learner.evaluate_classification_model(
                model=model,
                test_features=test_features.iloc[:30],
                test_target=pd.Series(true_labels)
            )
        
        # Check that the expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
        
        # Check that metrics are within valid ranges
        self.assertTrue(0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1")
        self.assertTrue(0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1")
        self.assertTrue(0 <= metrics['recall'] <= 1, "Recall should be between 0 and 1")
        self.assertTrue(0 <= metrics['f1'] <= 1, "F1 should be between 0 and 1")
        self.assertTrue(0 <= metrics['roc_auc'] <= 1, "ROC AUC should be between 0 and 1")
    
    def test_regression_evaluation_metrics(self):
        """Test that regression metrics are calculated correctly"""
        # Split data for testing
        train_features = self.features.iloc[:70]
        train_target = self.regression_target.iloc[:70]
        test_features = self.features.iloc[70:]
        test_target = self.regression_target.iloc[70:]
        
        # Mock the model's predict method
        model = MagicMock()
        # Create predictions with a known error pattern
        true_values = test_target.iloc[:30].values
        predictions = true_values + np.random.normal(0, 0.5, size=30)  # Add noise
        model.predict.return_value = predictions
        
        # Call the evaluation method with regression target
        with patch.object(self.learner, 'get_model', return_value=model):
            metrics = self.learner.evaluate_regression_model(
                model=model,
                test_features=test_features.iloc[:30],
                test_target=test_target.iloc[:30]
            )
        
        # Check that the expected metrics are present
        expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'explained_variance']
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
        
        # Check that metrics are within valid ranges or have reasonable values
        self.assertTrue(metrics['mse'] >= 0, "MSE should be non-negative")
        self.assertTrue(metrics['rmse'] >= 0, "RMSE should be non-negative")
        self.assertTrue(metrics['mae'] >= 0, "MAE should be non-negative")
        self.assertTrue(metrics['r2'] <= 1, "R² should be at most 1")
    
    def test_cross_validation(self):
        """Test that cross-validation is performed correctly"""
        # Create simple dataset for cross-validation
        X = self.features.iloc[:50]
        y = self.classification_target.iloc[:50]
        
        # Mock the cross_val_score function to return fixed scores
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.7, 0.75, 0.85, 0.82])
            
            # Call the cross-validation method
            cv_results = self.learner.cross_validate(
                X=X,
                y=y,
                model_type='classification',
                cv=5,
                scoring='accuracy'
            )
        
        # Check that cross-validation returns expected results
        self.assertIn('mean_score', cv_results, "CV results should include mean score")
        self.assertIn('std_score', cv_results, "CV results should include standard deviation")
        self.assertIn('all_scores', cv_results, "CV results should include all individual scores")
        
        # Check that the mean and std are calculated correctly
        expected_mean = 0.784
        expected_std = 0.0559
        self.assertAlmostEqual(cv_results['mean_score'], expected_mean, places=3)
        self.assertAlmostEqual(cv_results['std_score'], expected_std, places=3)
        
        # Check that all_scores contains the expected values
        np.testing.assert_almost_equal(cv_results['all_scores'], [0.8, 0.7, 0.75, 0.85, 0.82])
    
    def test_confusion_matrix_generation(self):
        """Test that confusion matrix is generated correctly"""
        # Mock model predictions
        y_true = np.array([0, 1, 0, 1, 0] * 10)  # 50 true labels
        y_pred = np.array([0, 1, 0, 0, 1] * 10)  # 50 predictions with some errors
        
        # Mock model
        model = MagicMock()
        model.predict.return_value = y_pred
        
        # Call the confusion matrix method
        with patch.object(self.learner, 'get_model', return_value=model):
            confusion_matrix = self.learner.get_confusion_matrix(
                model=model,
                test_features=self.features.iloc[:50],
                test_target=pd.Series(y_true)
            )
        
        # Expected confusion matrix for the given true labels and predictions
        expected_cm = np.array([[20, 10], [10, 10]])  # [[TN, FP], [FN, TP]]
        
        # Check that confusion matrix matches expected values
        np.testing.assert_array_equal(confusion_matrix, expected_cm)
    
    def test_feature_importance(self):
        """Test that feature importance is extracted correctly"""
        # Create a model with known feature importances
        with patch.object(self.model, 'feature_importances_', 
                         new=np.array([0.5, 0.3, 0.2])):
            
            # Call the feature importance method
            importances = self.learner.get_feature_importance(
                model=self.model,
                feature_names=self.features.columns
            )
        
        # Check that feature importances are returned in the correct format
        self.assertIsInstance(importances, dict, "Feature importances should be a dictionary")
        
        # Check that all features are included
        for feature in self.features.columns:
            self.assertIn(feature, importances, f"Missing feature importance for {feature}")
        
        # Check that importances match the expected values
        expected_importances = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        }
        for feature, importance in expected_importances.items():
            self.assertEqual(importances[feature], importance,
                            f"Incorrect importance for {feature}")
    
    def test_prediction_vs_actual_plot_data(self):
        """Test that prediction vs actual plot data is generated correctly"""
        # Mock model predictions
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3] * 10)  # 50 probability predictions
        
        # Create ground truth labels
        y_true = np.array([0, 1, 0, 1, 0] * 10)  # 50 true labels
        
        # Mock model
        model = MagicMock()
        model.predict_proba.return_value = np.column_stack((1 - predictions, predictions))
        
        # Call the method that generates plot data
        with patch.object(self.learner, 'get_model', return_value=model):
            plot_data = self.learner.get_prediction_vs_actual_data(
                model=model,
                test_features=self.features.iloc[:50],
                test_target=pd.Series(y_true)
            )
        
        # Check that the plot data has the expected format
        self.assertIn('predictions', plot_data, "Plot data should include predictions")
        self.assertIn('actuals', plot_data, "Plot data should include actual values")
        
        # Check that the arrays have the correct length
        self.assertEqual(len(plot_data['predictions']), 50, "Should have 50 predictions")
        self.assertEqual(len(plot_data['actuals']), 50, "Should have 50 actual values")
        
        # Check that the predictions match our mock values
        np.testing.assert_array_equal(plot_data['predictions'], predictions)
        
        # Check that the actuals match our ground truth
        np.testing.assert_array_equal(plot_data['actuals'], y_true)
    
    def test_roc_curve_data(self):
        """Test that ROC curve data is generated correctly"""
        # Mock model predictions
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3] * 10)  # 50 probability predictions
        
        # Create ground truth labels
        y_true = np.array([0, 1, 0, 1, 0] * 10)  # 50 true labels
        
        # Mock model
        model = MagicMock()
        model.predict_proba.return_value = np.column_stack((1 - predictions, predictions))
        
        # Mock sklearn.metrics.roc_curve
        with patch('sklearn.metrics.roc_curve') as mock_roc_curve:
            # Set return values for the mock
            fpr = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
            tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
            thresholds = np.array([1.0, 0.8, 0.6, 0.4, 0.0])
            mock_roc_curve.return_value = (fpr, tpr, thresholds)
            
            # Call the method that generates ROC curve data
            with patch.object(self.learner, 'get_model', return_value=model):
                roc_data = self.learner.get_roc_curve_data(
                    model=model,
                    test_features=self.features.iloc[:50],
                    test_target=pd.Series(y_true)
                )
        
        # Check that the ROC data has the expected format
        self.assertIn('fpr', roc_data, "ROC data should include false positive rates")
        self.assertIn('tpr', roc_data, "ROC data should include true positive rates")
        self.assertIn('thresholds', roc_data, "ROC data should include thresholds")
        
        # Check that the arrays match our mock values
        np.testing.assert_array_equal(roc_data['fpr'], fpr)
        np.testing.assert_array_equal(roc_data['tpr'], tpr)
        np.testing.assert_array_equal(roc_data['thresholds'], thresholds)


if __name__ == "__main__":
    unittest.main() 