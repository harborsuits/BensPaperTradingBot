#!/usr/bin/env python3
"""
Synthetic Testing Integration Core Tests

A focused test suite for the core functionality of the synthetic market testing integration.
This file tests the critical components without external dependencies:
- Event handling functionality
- Data analysis methods
- Configuration and setup

External dependencies are heavily mocked to allow tests to run in any environment.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import json
from datetime import datetime

# Import directly from the file to minimize dependencies
from trading_bot.autonomous.synthetic_testing_integration import (
    SyntheticTestingIntegration, get_synthetic_testing_integration
)
from trading_bot.event_system import Event, EventType


class TestSyntheticIntegrationCore(unittest.TestCase):
    """Test core functionality of the synthetic testing integration."""

    def setUp(self):
        """Set up test environment with mocks for all dependencies."""
        # Mock all external dependencies
        self.event_bus_mock = MagicMock()
        self.synthetic_generator_mock = MagicMock()
        self.correlated_generator_mock = MagicMock()
        self.ab_test_manager_mock = MagicMock()
        self.ab_test_analyzer_mock = MagicMock()
        self.approval_manager_mock = MagicMock()
        
        # Set up patchers
        self.event_bus_patcher = patch('trading_bot.event_system.EventBus', 
                                     return_value=self.event_bus_mock)
        self.synthetic_generator_patcher = patch(
            'trading_bot.autonomous.synthetic_market_generator.SyntheticMarketGenerator',
            return_value=self.synthetic_generator_mock
        )
        self.correlated_generator_patcher = patch(
            'trading_bot.autonomous.synthetic_market_generator_correlations.CorrelatedMarketGenerator',
            return_value=self.correlated_generator_mock
        )
        self.ab_test_manager_patcher = patch(
            'trading_bot.autonomous.synthetic_testing_integration.get_ab_test_manager',
            return_value=self.ab_test_manager_mock
        )
        self.ab_test_analyzer_patcher = patch(
            'trading_bot.autonomous.synthetic_testing_integration.get_ab_test_analyzer',
            return_value=self.ab_test_analyzer_mock
        )
        self.approval_manager_patcher = patch(
            'trading_bot.autonomous.synthetic_testing_integration.get_approval_workflow_manager',
            return_value=self.approval_manager_mock
        )
        
        # Start all patches
        self.event_bus_patcher.start()
        self.synthetic_generator_patcher.start()
        self.correlated_generator_patcher.start()
        self.ab_test_manager_patcher.start()
        self.ab_test_analyzer_patcher.start()
        self.approval_manager_patcher.start()
        
        # Create the integration instance
        self.integration = SyntheticTestingIntegration()

    def tearDown(self):
        """Clean up test environment."""
        # Stop all patches
        self.event_bus_patcher.stop()
        self.synthetic_generator_patcher.stop()
        self.correlated_generator_patcher.stop()
        self.ab_test_manager_patcher.stop()
        self.ab_test_analyzer_patcher.stop()
        self.approval_manager_patcher.stop()

    def test_initialization(self):
        """Test that the integration initializes correctly."""
        # Verify components are properly assigned
        self.assertEqual(self.integration.event_bus, self.event_bus_mock)
        self.assertEqual(self.integration.synthetic_generator, self.synthetic_generator_mock)
        self.assertEqual(self.integration.correlated_generator, self.correlated_generator_mock)
        self.assertEqual(self.integration.ab_test_manager, self.ab_test_manager_mock)
        self.assertEqual(self.integration.ab_test_analyzer, self.ab_test_analyzer_mock)
        self.assertEqual(self.integration.approval_manager, self.approval_manager_mock)
        
        # Verify default configuration values
        self.assertEqual(self.integration.default_test_days, 252)
        self.assertIsInstance(self.integration.default_symbols, list)
        self.assertIsInstance(self.integration.default_regimes, list)
    
    def test_event_registration(self):
        """Test that event handlers are registered correctly."""
        # Verify that the event handlers are registered
        # We expect at least two event registrations:
        # 1. For test creation events
        # 2. For approval request creation events
        expected_calls = [
            call("ab_test_created", self.integration._handle_test_created),
            call(EventType.APPROVAL_REQUEST_CREATED, self.integration._handle_approval_request_created)
        ]
        
        # Check that all expected calls were made
        self.event_bus_mock.register.assert_has_calls(expected_calls, any_order=True)
    
    def test_handle_test_created(self):
        """Test handling of test creation events."""
        # Create a mock test with synthetic testing enabled
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"use_synthetic_testing": True}
        
        # Configure the test manager to return our mock
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create a test event
        test_event = MagicMock()
        test_event.data = {"test_id": "test123"}
        
        # Replace add_synthetic_testing_to_test with a mock
        self.integration.add_synthetic_testing_to_test = MagicMock()
        
        # Call the event handler
        self.integration._handle_test_created(test_event)
        
        # Verify that the test manager was called correctly
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Verify that add_synthetic_testing_to_test was called
        self.integration.add_synthetic_testing_to_test.assert_called_once_with(mock_test)
    
    def test_handle_test_created_no_synthetic(self):
        """Test handling of test creation when synthetic testing is disabled."""
        # Create a mock test with synthetic testing disabled
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"use_synthetic_testing": False}
        
        # Configure the test manager to return our mock
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create a test event
        test_event = MagicMock()
        test_event.data = {"test_id": "test123"}
        
        # Replace add_synthetic_testing_to_test with a mock
        self.integration.add_synthetic_testing_to_test = MagicMock()
        
        # Call the event handler
        self.integration._handle_test_created(test_event)
        
        # Verify that the test manager was called correctly
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Verify that add_synthetic_testing_to_test was NOT called
        self.integration.add_synthetic_testing_to_test.assert_not_called()
    
    def test_handle_approval_request_created(self):
        """Test handling of approval request creation events."""
        # Create mock request and test
        mock_request = MagicMock()
        mock_request.request_id = "req123"
        mock_request.test_id = "test123"
        
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"synthetic_testing_completed": False}
        
        # Configure managers to return mocks
        self.approval_manager_mock.get_request.return_value = mock_request
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create event
        event = MagicMock()
        event.data = {
            "request_id": "req123",
            "test_id": "test123"
        }
        
        # Mock the enhancement method
        self.integration.enhance_approval_request_with_synthetic_data = MagicMock()
        
        # Call handler
        self.integration._handle_approval_request_created(event)
        
        # Verify correct workflow
        self.approval_manager_mock.get_request.assert_called_once_with("req123")
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        self.integration.enhance_approval_request_with_synthetic_data.assert_called_once_with(
            mock_request, mock_test
        )
    
    def test_handle_approval_request_already_processed(self):
        """Test handling of approval request when synthetic testing is already completed."""
        # Create mock request and test with synthetic testing already completed
        mock_request = MagicMock()
        mock_request.request_id = "req123"
        mock_request.test_id = "test123"
        
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"synthetic_testing_completed": True}
        
        # Configure managers to return mocks
        self.approval_manager_mock.get_request.return_value = mock_request
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create event
        event = MagicMock()
        event.data = {
            "request_id": "req123",
            "test_id": "test123"
        }
        
        # Mock the enhancement method
        self.integration.enhance_approval_request_with_synthetic_data = MagicMock()
        
        # Call handler
        self.integration._handle_approval_request_created(event)
        
        # Verify that enhancement was NOT called
        self.integration.enhance_approval_request_with_synthetic_data.assert_not_called()
    
    def test_compare_backtest_results(self):
        """Test comparison logic between variant test results."""
        # Set up test inputs where variant B is better
        variant_a_results = {
            "metrics": {
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.2,
                "win_rate": 0.5,
                "profit_factor": 1.5
            }
        }
        
        variant_b_results = {
            "metrics": {
                "sharpe_ratio": 1.5,  # Better
                "max_drawdown": -0.15,  # Better (less negative)
                "win_rate": 0.6,  # Better
                "profit_factor": 1.8  # Better
            }
        }
        
        # Run the comparison
        result = self.integration._compare_backtest_results(
            variant_a_results, variant_b_results
        )
        
        # Verify results structure
        self.assertIn("b_is_better", result)
        self.assertIn("differences", result)
        self.assertIn("relative_improvements", result)
        self.assertIn("confidence_score", result)
        
        # Verify that B is correctly identified as better
        self.assertTrue(result["b_is_better"])
        
        # Verify difference calculations
        differences = result["differences"]
        self.assertEqual(differences["sharpe_ratio"], 0.5)
        self.assertEqual(differences["max_drawdown"], 0.05)  # -0.15 - (-0.2) = 0.05
        self.assertEqual(differences["win_rate"], 0.1)
        self.assertEqual(differences["profit_factor"], 0.3)
        
        # Verify confidence score is in valid range (0-1)
        self.assertGreaterEqual(result["confidence_score"], 0.0)
        self.assertLessEqual(result["confidence_score"], 1.0)
    
    def test_generate_regime_summary(self):
        """Test generation of regime summary from test results."""
        # Create test regime results with B better in 3 of 4 regimes
        regime_results = {
            "bullish": {
                "comparison": {
                    "b_is_better": True,
                    "confidence_score": 0.8,
                    "differences": {
                        "sharpe_ratio": 0.5,
                        "max_drawdown": 0.05
                    }
                }
            },
            "bearish": {
                "comparison": {
                    "b_is_better": True,
                    "confidence_score": 0.7,
                    "differences": {
                        "sharpe_ratio": 0.3,
                        "max_drawdown": 0.03
                    }
                }
            },
            "volatile": {
                "comparison": {
                    "b_is_better": True,
                    "confidence_score": 0.6,
                    "differences": {
                        "sharpe_ratio": 0.2,
                        "max_drawdown": 0.02
                    }
                }
            },
            "sideways": {
                "comparison": {
                    "b_is_better": False,
                    "confidence_score": 0.0,
                    "differences": {
                        "sharpe_ratio": -0.1,
                        "max_drawdown": -0.01
                    }
                }
            }
        }
        
        # Generate summary
        summary = self.integration._generate_regime_summary(regime_results)
        
        # Verify summary structure
        self.assertIn("total_regimes", summary)
        self.assertIn("regimes_b_better", summary)
        self.assertIn("promote_b", summary)
        self.assertIn("confidence", summary)
        self.assertIn("regime_specific", summary)
        
        # Verify values
        self.assertEqual(summary["total_regimes"], 4)
        self.assertEqual(summary["regimes_b_better"], 3)
        self.assertTrue(summary["promote_b"])  # 3/4 > 0.75
        
        # Verify regime-specific recommendations
        regime_specific = summary["regime_specific"]
        self.assertEqual(len(regime_specific), 4)
        self.assertEqual(regime_specific["bullish"]["recommended_variant"], "B")
        self.assertEqual(regime_specific["sideways"]["recommended_variant"], "A")
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation based on regime results."""
        # Create test regime results with B better in 3 of 4 regimes
        regime_results = {
            "bullish": {
                "comparison": {
                    "b_is_better": True,
                    "confidence_score": 0.9
                }
            },
            "bearish": {
                "comparison": {
                    "b_is_better": True,
                    "confidence_score": 0.8
                }
            },
            "volatile": {
                "comparison": {
                    "b_is_better": True,
                    "confidence_score": 0.7
                }
            },
            "sideways": {
                "comparison": {
                    "b_is_better": False,
                    "confidence_score": 0.0
                }
            }
        }
        
        # Calculate confidence score
        score = self.integration._calculate_confidence_score(regime_results)
        
        # Verify score is in valid range (0-1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Score should be influenced by both breadth (3/4 regimes) and depth
        # (average confidence of 0.8 for regimes where B is better)
        expected_score = (3/4) * ((0.9 + 0.8 + 0.7) / 3)
        self.assertAlmostEqual(score, expected_score, delta=0.01)
        
        # For a case with no regimes where B is better, score should be 0
        no_b_better_results = {
            "bullish": {
                "comparison": {
                    "b_is_better": False,
                    "confidence_score": 0.0
                }
            },
            "bearish": {
                "comparison": {
                    "b_is_better": False,
                    "confidence_score": 0.0
                }
            }
        }
        score = self.integration._calculate_confidence_score(no_b_better_results)
        self.assertEqual(score, 0.0)
    
    def test_singleton_accessor(self):
        """Test the singleton accessor function."""
        # Access the singleton
        with patch('trading_bot.autonomous.synthetic_testing_integration.SyntheticTestingIntegration',
                  return_value=MagicMock()) as mock_init:
            # First call should create a new instance
            instance1 = get_synthetic_testing_integration()
            mock_init.assert_called_once()
            
            # Second call should return the same instance
            instance2 = get_synthetic_testing_integration()
            # Still only called once
            mock_init.assert_called_once()
            
            # Both variables should reference the same instance
            self.assertEqual(instance1, instance2)


if __name__ == "__main__":
    unittest.main()
