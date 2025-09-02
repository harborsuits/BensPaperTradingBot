#!/usr/bin/env python3
"""
A/B Testing Framework Test Suite

This module provides comprehensive unit tests for the A/B Testing Framework
components, including:
- Core data structures (TestVariant, TestMetrics, ABTest)
- ABTestManager for lifecycle handling
- ABTestAnalyzer for statistical analysis
- ABTestingIntegration for system integration

It uses synthetic data and mocks to avoid external dependencies,
following our pattern of thorough code-level verification.
"""

import os
import json
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Import A/B testing components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)
from trading_bot.autonomous.ab_testing_manager import (
    ABTestManager, ABTestEventType
)
from trading_bot.autonomous.ab_testing_analysis import (
    SignificanceTest, ConfidenceInterval, ABTestAnalyzer
)
from trading_bot.autonomous.ab_testing_integration import (
    ABTestingIntegration
)

# Import event system for mocking
from trading_bot.event_system import EventBus, Event


class TestABTestCore(unittest.TestCase):
    """Test core A/B testing data structures."""
    
    def test_test_variant_creation(self):
        """Test TestVariant creation and methods."""
        # Create a test variant
        variant = TestVariant(
            strategy_id="test_strategy",
            version_id="v1.0",
            name="Test Variant",
            parameters={"param1": 10, "param2": "value"}
        )
        
        # Check basic properties
        self.assertEqual(variant.strategy_id, "test_strategy")
        self.assertEqual(variant.version_id, "v1.0")
        self.assertEqual(variant.name, "Test Variant")
        self.assertEqual(variant.parameters, {"param1": 10, "param2": "value"})
        self.assertIsInstance(variant.variant_id, str)
        self.assertIsInstance(variant.metrics, dict)
        self.assertEqual(len(variant.metrics), 0)
        self.assertIsInstance(variant.trade_history, list)
        self.assertEqual(len(variant.trade_history), 0)
        
        # Test update_metrics
        variant.update_metrics({
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "max_drawdown": -0.15
        })
        
        self.assertEqual(variant.metrics["sharpe_ratio"], 1.5)
        self.assertEqual(variant.metrics["win_rate"], 0.6)
        self.assertEqual(variant.metrics["max_drawdown"], -0.15)
        
        # Test add_trade
        trade = {
            "trade_id": "trade1",
            "entry_time": datetime.now().isoformat(),
            "exit_time": (datetime.now() + timedelta(hours=1)).isoformat(),
            "profit": 100.0,
            "return": 0.05
        }
        
        variant.add_trade(trade)
        self.assertEqual(len(variant.trade_history), 1)
        self.assertEqual(variant.trade_history[0]["trade_id"], "trade1")
        
        # Test to_dict and from_dict
        variant_dict = variant.to_dict()
        self.assertIsInstance(variant_dict, dict)
        
        new_variant = TestVariant.from_dict(variant_dict)
        self.assertEqual(new_variant.strategy_id, variant.strategy_id)
        self.assertEqual(new_variant.version_id, variant.version_id)
        self.assertEqual(new_variant.name, variant.name)
        self.assertEqual(new_variant.parameters, variant.parameters)
        self.assertEqual(new_variant.metrics, variant.metrics)
    
    def test_ab_test_creation(self):
        """Test ABTest creation and lifecycle methods."""
        # Create variants
        variant_a = TestVariant(
            strategy_id="test_strategy",
            version_id="v1.0",
            name="Baseline",
            parameters={"param1": 10}
        )
        
        variant_b = TestVariant(
            strategy_id="test_strategy",
            version_id="v2.0",
            name="Experiment",
            parameters={"param1": 20}
        )
        
        # Create test
        test = ABTest(
            name="Test Comparison",
            variant_a=variant_a,
            variant_b=variant_b,
            config={
                "duration_days": 30,
                "confidence_level": 0.95,
                "metrics_to_compare": ["sharpe_ratio", "win_rate"]
            },
            description="Testing parameter changes",
            metadata={"source": "unit_test"}
        )
        
        # Check basic properties
        self.assertEqual(test.name, "Test Comparison")
        self.assertEqual(test.description, "Testing parameter changes")
        self.assertEqual(test.variant_a.name, "Baseline")
        self.assertEqual(test.variant_b.name, "Experiment")
        self.assertEqual(test.status, TestStatus.CREATED)
        self.assertIsInstance(test.created_at, datetime)
        self.assertIsNone(test.started_at)
        self.assertIsNone(test.completed_at)
        
        # Test lifecycle methods
        test.start_test()
        self.assertEqual(test.status, TestStatus.RUNNING)
        self.assertIsInstance(test.started_at, datetime)
        
        test.stop_test("Manual stop")
        self.assertEqual(test.status, TestStatus.STOPPED)
        self.assertEqual(test.conclusion, "Manual stop")
        
        # Test restart
        test.start_test()
        self.assertEqual(test.status, TestStatus.RUNNING)
        
        # Add metrics to variants
        test.variant_a.update_metrics({
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "max_drawdown": -0.2
        })
        
        test.variant_b.update_metrics({
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "max_drawdown": -0.15
        })
        
        # Complete test
        test.complete_test()
        self.assertEqual(test.status, TestStatus.COMPLETED)
        self.assertIsInstance(test.completed_at, datetime)
        
        # Check results
        self.assertEqual(test.winner, "B")
        self.assertIn("Variant B", test.conclusion)
        self.assertTrue(test.should_promote_variant_b())
        
        # Test to_dict and from_dict
        test_dict = test.to_dict()
        self.assertIsInstance(test_dict, dict)
        
        new_test = ABTest.from_dict(test_dict)
        self.assertEqual(new_test.test_id, test.test_id)
        self.assertEqual(new_test.name, test.name)
        self.assertEqual(new_test.status, test.status)
        self.assertEqual(new_test.winner, test.winner)
        self.assertEqual(new_test.variant_a.name, test.variant_a.name)
        self.assertEqual(new_test.variant_b.name, test.variant_b.name)


class TestABTestManager(unittest.TestCase):
    """Test the A/B Test Manager component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock event bus
        self.mock_event_bus = MagicMock()
        
        # Patch the EventBus to return our mock
        self.event_bus_patch = patch('trading_bot.autonomous.ab_testing_manager.EventBus')
        self.mock_event_bus_class = self.event_bus_patch.start()
        self.mock_event_bus_class.return_value = self.mock_event_bus
        
        # Patch the LifecycleManager and CorrelationRegimeDetector
        self.lifecycle_patch = patch('trading_bot.autonomous.ab_testing_manager.get_lifecycle_manager')
        self.mock_lifecycle = self.lifecycle_patch.start()
        
        # Create the ABTestManager with our temp directory
        self.manager = ABTestManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Stop patches
        self.event_bus_patch.stop()
        self.lifecycle_patch.stop()
    
    def test_create_and_manage_test(self):
        """Test creating and managing an A/B test."""
        # Create test variants
        variant_a = TestVariant(
            strategy_id="test_strategy",
            version_id="v1.0",
            name="Baseline",
            parameters={"param1": 10}
        )
        
        variant_b = TestVariant(
            strategy_id="test_strategy",
            version_id="v2.0",
            name="Experiment",
            parameters={"param1": 20}
        )
        
        # Create test
        test = self.manager.create_test(
            name="Test Comparison",
            variant_a=variant_a,
            variant_b=variant_b,
            config={
                "duration_days": 30,
                "confidence_level": 0.95,
                "metrics_to_compare": ["sharpe_ratio", "win_rate"]
            },
            description="Testing parameter changes",
            metadata={"source": "unit_test"}
        )
        
        # Check that test was created and stored
        self.assertIsInstance(test, ABTest)
        self.assertEqual(test.name, "Test Comparison")
        
        # Check that test file was created
        test_file = os.path.join(self.temp_dir, f"{test.test_id}.json")
        self.assertTrue(os.path.exists(test_file))
        
        # Check that test can be retrieved
        retrieved_test = self.manager.get_test(test.test_id)
        self.assertEqual(retrieved_test.test_id, test.test_id)
        self.assertEqual(retrieved_test.name, test.name)
        
        # Test starting the test
        self.manager.start_test(test.test_id)
        retrieved_test = self.manager.get_test(test.test_id)
        self.assertEqual(retrieved_test.status, TestStatus.RUNNING)
        
        # Test stopping the test
        self.manager.stop_test(test.test_id, "Manual stop")
        retrieved_test = self.manager.get_test(test.test_id)
        self.assertEqual(retrieved_test.status, TestStatus.STOPPED)
        
        # Test starting again
        self.manager.start_test(test.test_id)
        
        # Add metrics to variants
        event_data = {
            "strategy_id": "test_strategy",
            "version_id": "v1.0",
            "performance": {
                "sharpe_ratio": 1.2,
                "win_rate": 0.55,
                "max_drawdown": -0.2
            }
        }
        
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = event_data
        
        # Call the handler directly
        self.manager._handle_performance_updated(mock_event)
        
        # Update variant B
        event_data["version_id"] = "v2.0"
        event_data["performance"] = {
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "max_drawdown": -0.15
        }
        mock_event.data = event_data
        self.manager._handle_performance_updated(mock_event)
        
        # Test completing the test
        self.manager.complete_test(test.test_id)
        retrieved_test = self.manager.get_test(test.test_id)
        self.assertEqual(retrieved_test.status, TestStatus.COMPLETED)
        self.assertEqual(retrieved_test.winner, "B")
        
        # Test getting test summary
        summary = self.manager.get_test_summary(test.test_id)
        self.assertEqual(summary["name"], "Test Comparison")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["winner"], "B")
        
        # Test deleting the test
        self.manager.delete_test(test.test_id)
        self.assertIsNone(self.manager.get_test(test.test_id))
        self.assertFalse(os.path.exists(test_file))
    
    def test_scheduler_loop(self):
        """Test the scheduler loop behavior."""
        # Create and start a test with a very short duration
        variant_a = TestVariant(
            strategy_id="test_strategy",
            version_id="v1.0",
            name="Baseline"
        )
        
        variant_b = TestVariant(
            strategy_id="test_strategy",
            version_id="v2.0",
            name="Experiment"
        )
        
        test = self.manager.create_test(
            name="Short Test",
            variant_a=variant_a,
            variant_b=variant_b,
            config={"duration_days": 0.001}  # Very short duration (about 1.5 minutes)
        )
        
        # Start the test and set started_at to a time in the past
        self.manager.start_test(test.test_id)
        retrieved_test = self.manager.get_test(test.test_id)
        
        # Manually adjust started_at to a time in the past
        retrieved_test.started_at = datetime.now() - timedelta(days=1)
        self.manager._save_test(retrieved_test)
        
        # Mock the complete_test method to track calls
        original_complete_test = self.manager.complete_test
        self.manager.complete_test = MagicMock(side_effect=original_complete_test)
        
        # Run the scheduler loop once
        with patch('time.sleep'):  # Patch sleep to avoid waiting
            self.manager._check_test_completion()
        
        # Check that complete_test was called
        self.manager.complete_test.assert_called_once_with(test.test_id)
        
        # Reset the mock
        self.manager.complete_test = original_complete_test


class TestABTestAnalysis(unittest.TestCase):
    """Test the A/B Test Analysis component."""
    
    def test_significance_test_t_test(self):
        """Test the t-test implementation."""
        # Create synthetic data where B is clearly better
        data_a = np.random.normal(0.01, 0.02, 100)  # Mean 1%, SD 2%
        data_b = np.random.normal(0.02, 0.02, 100)  # Mean 2%, SD 2%
        
        # Run t-test
        result = SignificanceTest.t_test(data_a, data_b)
        
        # Check result structure
        self.assertIn("is_significant", result)
        self.assertIn("p_value", result)
        self.assertIn("test_statistic", result)
        self.assertIn("confidence_level", result)
        
        # With this synthetic data, B should be significantly better
        self.assertTrue(result["is_significant"])
        
        # Test with insufficient data
        result = SignificanceTest.t_test([0.01], [0.02])
        self.assertFalse(result["is_significant"])
        self.assertIn("error", result)
    
    def test_significance_test_bootstrap(self):
        """Test the bootstrap resampling implementation."""
        # Create synthetic data where B is clearly better
        data_a = np.random.normal(0.01, 0.02, 100)  # Mean 1%, SD 2%
        data_b = np.random.normal(0.03, 0.02, 100)  # Mean 3%, SD 2%
        
        # Run bootstrap test
        result = SignificanceTest.bootstrap(
            data_a, data_b, statistic='mean', n_iterations=100
        )
        
        # Check result structure
        self.assertIn("is_significant", result)
        self.assertIn("p_value", result)
        self.assertIn("confidence_level", result)
        self.assertIn("original_diff", result)
        self.assertIn("confidence_interval", result)
        
        # With this synthetic data, B should be significantly better
        self.assertTrue(result["is_significant"])
        self.assertGreater(result["original_diff"], 0)
        
        # Test with insufficient data
        result = SignificanceTest.bootstrap([0.01], [0.02])
        self.assertFalse(result["is_significant"])
        self.assertIn("error", result)
    
    def test_confidence_intervals(self):
        """Test confidence interval calculations."""
        # Test mean confidence interval
        data = np.random.normal(0.02, 0.03, 100)  # Mean 2%, SD 3%
        mean, lower, upper = ConfidenceInterval.mean_confidence_interval(data)
        
        self.assertAlmostEqual(mean, np.mean(data), places=6)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        
        # Test Sharpe ratio confidence interval
        returns = np.random.normal(0.01, 0.02, 100)  # Mean 1%, SD 2%
        sharpe, lower, upper = ConfidenceInterval.sharpe_ratio_confidence_interval(returns)
        
        expected_sharpe = np.mean(returns) / np.std(returns, ddof=1)
        self.assertAlmostEqual(sharpe, expected_sharpe, places=6)
        self.assertLess(lower, sharpe)
        self.assertGreater(upper, sharpe)
        
        # Test win rate confidence interval
        wins = 60
        total = 100
        win_rate, lower, upper = ConfidenceInterval.win_rate_confidence_interval(wins, total)
        
        self.assertEqual(win_rate, 0.6)
        self.assertLess(lower, win_rate)
        self.assertGreater(upper, win_rate)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
    
    def test_ab_test_analyzer(self):
        """Test the AB test analyzer."""
        # Create variants
        variant_a = TestVariant(
            strategy_id="test_strategy",
            version_id="v1.0",
            name="Baseline"
        )
        
        variant_b = TestVariant(
            strategy_id="test_strategy",
            version_id="v2.0",
            name="Experiment"
        )
        
        # Add trade history
        # Variant A: mean return 1%, SD 2%
        for i in range(100):
            ret = np.random.normal(0.01, 0.02)
            variant_a.add_trade({
                "trade_id": f"a{i}",
                "entry_time": datetime.now().isoformat(),
                "exit_time": (datetime.now() + timedelta(hours=1)).isoformat(),
                "profit": 100 * ret,
                "return": ret
            })
        
        # Variant B: mean return 2%, SD 2%
        for i in range(100):
            ret = np.random.normal(0.02, 0.02)
            variant_b.add_trade({
                "trade_id": f"b{i}",
                "entry_time": datetime.now().isoformat(),
                "exit_time": (datetime.now() + timedelta(hours=1)).isoformat(),
                "profit": 100 * ret,
                "return": ret
            })
        
        # Add metrics
        variant_a.update_metrics({
            "sharpe_ratio": 0.5,
            "win_rate": 0.55,
            "max_drawdown": -0.2,
            "sortino_ratio": 0.7
        })
        
        variant_b.update_metrics({
            "sharpe_ratio": 1.0,
            "win_rate": 0.6,
            "max_drawdown": -0.15,
            "sortino_ratio": 1.2
        })
        
        # Create and complete test
        test = ABTest(
            name="Test Analysis",
            variant_a=variant_a,
            variant_b=variant_b
        )
        test.start_test()
        test.complete_test()
        
        # Create analyzer
        analyzer = ABTestAnalyzer()
        
        # Analyze test
        results = analyzer.analyze_test(test)
        
        # Check result structure
        self.assertIn("test_id", results)
        self.assertIn("name", results)
        self.assertIn("significance_tests", results)
        self.assertIn("confidence_intervals", results)
        self.assertIn("recommendation", results)
        
        # With our synthetic data, variant B should be recommended
        recommendation = results["recommendation"]
        self.assertTrue(recommendation["promote_variant_b"])
        self.assertIn("explanation", recommendation)


class TestABTestingIntegration(unittest.TestCase):
    """Test the A/B Testing Integration component."""
    
    def setUp(self):
        """Set up test environment."""
        # Patch all external components
        self.ab_test_manager_patch = patch('trading_bot.autonomous.ab_testing_integration.get_ab_test_manager')
        self.ab_test_analyzer_patch = patch('trading_bot.autonomous.ab_testing_integration.get_ab_test_analyzer')
        self.event_bus_patch = patch('trading_bot.autonomous.ab_testing_integration.EventBus')
        self.lifecycle_manager_patch = patch('trading_bot.autonomous.ab_testing_integration.get_lifecycle_manager')
        self.autonomous_engine_patch = patch('trading_bot.autonomous.ab_testing_integration.get_autonomous_engine')
        
        # Start patches
        self.mock_ab_test_manager = self.ab_test_manager_patch.start()
        self.mock_ab_test_analyzer = self.ab_test_analyzer_patch.start()
        self.mock_event_bus_class = self.event_bus_patch.start()
        self.mock_lifecycle_manager = self.lifecycle_manager_patch.start()
        self.mock_autonomous_engine = self.autonomous_engine_patch.start()
        
        # Create mock instances
        self.mock_ab_test_manager_instance = MagicMock()
        self.mock_ab_test_analyzer_instance = MagicMock()
        self.mock_event_bus = MagicMock()
        self.mock_lifecycle_manager_instance = MagicMock()
        self.mock_autonomous_engine_instance = MagicMock()
        
        # Configure mocks
        self.mock_ab_test_manager.return_value = self.mock_ab_test_manager_instance
        self.mock_ab_test_analyzer.return_value = self.mock_ab_test_analyzer_instance
        self.mock_event_bus_class.return_value = self.mock_event_bus
        self.mock_lifecycle_manager.return_value = self.mock_lifecycle_manager_instance
        self.mock_autonomous_engine.return_value = self.mock_autonomous_engine_instance
        
        # Create integration instance
        self.integration = ABTestingIntegration()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.ab_test_manager_patch.stop()
        self.ab_test_analyzer_patch.stop()
        self.event_bus_patch.stop()
        self.lifecycle_manager_patch.stop()
        self.autonomous_engine_patch.stop()
    
    def test_initialization(self):
        """Test integration initialization."""
        # Check that the integration initialized all components
        self.mock_ab_test_manager.assert_called_once()
        self.mock_ab_test_analyzer.assert_called_once()
        self.mock_event_bus_class.assert_called_once()
        self.mock_lifecycle_manager.assert_called_once()
        self.mock_autonomous_engine.assert_called_once()
        
        # Check that event handlers were registered
        self.assertGreater(self.mock_event_bus.register.call_count, 0)
        
        # Check that AB test manager was started
        self.mock_ab_test_manager_instance.start.assert_called_once()
    
    def test_test_completed_handler(self):
        """Test the test completed event handler."""
        # Create mock test
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.name = "Test Handler"
        mock_test.variant_b = MagicMock()
        
        # Configure manager to return our mock test
        self.mock_ab_test_manager_instance.get_test.return_value = mock_test
        
        # Configure analyzer to return recommendation
        self.mock_ab_test_analyzer_instance.analyze_test.return_value = {
            "recommendation": {
                "promote_variant_b": True,
                "confidence": "high",
                "explanation": "B is better",
                "regime_switching": {
                    "recommended": False
                }
            }
        }
        
        # Create event with test completed data
        event = MagicMock()
        event.data = {
            "test_id": "test123",
            "name": "Test Handler",
            "result": {
                "winner": "B",
                "conclusion": "B is better"
            }
        }
        
        # Call handler
        self.integration._handle_test_completed(event)
        
        # Check that test was retrieved
        self.mock_ab_test_manager_instance.get_test.assert_called_once_with("test123")
        
        # Check that test was analyzed
        self.mock_ab_test_analyzer_instance.analyze_test.assert_called_once_with(mock_test)
        
        # Check that events were emitted
        self.assertGreater(self.mock_event_bus.emit.call_count, 0)
    
    def test_create_test_from_optimization(self):
        """Test creating a test from optimization results."""
        # Configure lifecycle manager
        mock_version = MagicMock()
        mock_version.parameters = {"param1": 10}
        self.mock_lifecycle_manager_instance.get_version.return_value = mock_version
        
        # Configure AB test manager
        mock_test = MagicMock()
        self.mock_ab_test_manager_instance.create_test.return_value = mock_test
        
        # Call method
        result = self.integration.create_test_from_optimization(
            strategy_id="test_strategy",
            original_version_id="v1.0",
            new_parameters={"param1": 20},
            job_id="job123",
            auto_start=True
        )
        
        # Check that lifecycle manager was queried
        self.mock_lifecycle_manager_instance.get_version.assert_called_once_with(
            "test_strategy", "v1.0"
        )
        
        # Check that test was created
        self.mock_ab_test_manager_instance.create_test.assert_called_once()
        args, kwargs = self.mock_ab_test_manager_instance.create_test.call_args
        self.assertEqual(kwargs["name"], "Optimization Test: test_strategy")
        
        # Check that test was started
        self.mock_ab_test_manager_instance.start_test.assert_called_once_with(mock_test.test_id)
        
        # Check return value
        self.assertEqual(result, mock_test)
    
    def test_analyze_and_apply_test_results(self):
        """Test analyzing and applying test results."""
        # Configure AB test manager
        mock_test = MagicMock()
        mock_test.status = TestStatus.COMPLETED
        mock_test.variant_b = MagicMock()
        self.mock_ab_test_manager_instance.get_test.return_value = mock_test
        
        # Configure analyzer
        self.mock_ab_test_analyzer_instance.analyze_test.return_value = {
            "recommendation": {
                "promote_variant_b": True,
                "confidence": "high",
                "explanation": "B is better"
            }
        }
        
        # Call method
        result = self.integration.analyze_and_apply_test_results("test123")
        
        # Check that test was retrieved
        self.mock_ab_test_manager_instance.get_test.assert_called_once_with("test123")
        
        # Check that test was analyzed
        self.mock_ab_test_analyzer_instance.analyze_test.assert_called_once_with(mock_test)
        
        # Check that promotion was attempted
        self.mock_lifecycle_manager_instance.get_version.assert_called_once()
        
        # Check return value
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
