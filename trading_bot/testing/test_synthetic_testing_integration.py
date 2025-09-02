#!/usr/bin/env python3
"""
Synthetic Testing Integration Test Suite

This module provides comprehensive tests for the Synthetic Market Testing Integration,
ensuring proper integration with the A/B Testing Framework and Approval Workflow.

Tests cover:
- Unit tests for core functionality
- Integration tests with approval workflow
- Simulation tests with different market regimes
"""

import os
import json
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import numpy as np
import pandas as pd

# Import synthetic market components
from trading_bot.autonomous.synthetic_market_generator import (
    SyntheticMarketGenerator, MarketRegimeType
)
from trading_bot.autonomous.synthetic_market_generator_correlations import (
    CorrelatedMarketGenerator, CorrelationStructure
)

# Import synthetic testing integration
from trading_bot.autonomous.synthetic_testing_integration import (
    SyntheticTestingIntegration, get_synthetic_testing_integration
)

# Import A/B testing components
from trading_bot.autonomous.ab_testing_core import (
    ABTest, TestVariant, TestMetrics, TestStatus
)

# Import approval workflow components
from trading_bot.autonomous.approval_workflow import (
    ApprovalStatus, ApprovalRequest
)

# Import event system components
from trading_bot.event_system import EventBus, Event, EventType


class TestSyntheticTestingCore(unittest.TestCase):
    """Test core functionality of the synthetic testing integration."""

    def setUp(self):
        """Set up test environment with mocks."""
        # Mock dependencies
        self.event_bus_mock = MagicMock()
        self.synthetic_generator_mock = MagicMock()
        self.correlated_generator_mock = MagicMock()
        self.ab_test_manager_mock = MagicMock()
        self.ab_test_analyzer_mock = MagicMock()
        self.approval_manager_mock = MagicMock()
        
        # Create patchers
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
        # Check that the integration has all required components
        self.assertEqual(self.integration.event_bus, self.event_bus_mock)
        self.assertEqual(self.integration.synthetic_generator, self.synthetic_generator_mock)
        self.assertEqual(self.integration.correlated_generator, self.correlated_generator_mock)
        self.assertEqual(self.integration.ab_test_manager, self.ab_test_manager_mock)
        self.assertEqual(self.integration.ab_test_analyzer, self.ab_test_analyzer_mock)
        self.assertEqual(self.integration.approval_manager, self.approval_manager_mock)
        
        # Check default configuration
        self.assertEqual(self.integration.default_test_days, 252)
        self.assertIsInstance(self.integration.default_symbols, list)
        self.assertIsInstance(self.integration.default_regimes, list)
    
    def test_event_registration(self):
        """Test that the integration registers event handlers correctly."""
        # Check that register was called for test creation events
        self.event_bus_mock.register.assert_any_call(
            "ab_test_created",
            self.integration._handle_test_created
        )
        
        # Check that register was called for approval request creation events
        self.event_bus_mock.register.assert_any_call(
            EventType.APPROVAL_REQUEST_CREATED,
            self.integration._handle_approval_request_created
        )
    
    def test_generate_regime_specific_data(self):
        """Test generation of regime-specific market data."""
        # Mock the correlated generator to return a test dataframe
        test_df = pd.DataFrame({'price': [100, 101, 102]})
        market_data = {'BTC': test_df, 'ETH': test_df}
        self.correlated_generator_mock.generate_correlated_markets.return_value = market_data
        
        # Test bullish regime
        result = self.integration._generate_regime_specific_data(
            symbols=['BTC', 'ETH'],
            days=30,
            regime=MarketRegimeType.BULLISH
        )
        
        # Check that correlated generator was called correctly
        self.correlated_generator_mock.generate_correlated_markets.assert_called_once()
        
        # Check that synthetic generator applied momentum effect for bullish regime
        self.synthetic_generator_mock.apply_momentum_effect.assert_called()
        
        # Reset mocks for next test
        self.correlated_generator_mock.reset_mock()
        self.synthetic_generator_mock.reset_mock()
        
        # Test bearish regime
        self.correlated_generator_mock.generate_correlated_markets.return_value = market_data
        result = self.integration._generate_regime_specific_data(
            symbols=['BTC', 'ETH'],
            days=30,
            regime=MarketRegimeType.BEARISH
        )
        
        # Check that panic selling effect was applied for bearish regime
        self.synthetic_generator_mock.apply_panic_selling_effect.assert_called()
    
    def test_backtest_strategy(self):
        """Test strategy backtesting simulation."""
        # Create test market data
        test_df = pd.DataFrame({'price': [100, 101, 102]})
        market_data = {'BTC': test_df, 'ETH': test_df}
        
        # Run backtest
        result = self.integration._backtest_strategy(
            strategy_id="test_strategy",
            version_id="v1.0",
            market_data=market_data
        )
        
        # Check result structure
        self.assertIn("strategy_id", result)
        self.assertIn("version_id", result)
        self.assertIn("metrics", result)
        self.assertIn("trades", result)
        
        # Check metrics
        metrics = result["metrics"]
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_factor", metrics)
        self.assertIn("trade_count", metrics)
    
    def test_compare_backtest_results(self):
        """Test comparison of backtest results between variants."""
        # Create test results for A and B
        variant_a_results = {
            "strategy_id": "test_strategy",
            "version_id": "v1.0",
            "metrics": {
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.2,
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "trade_count": 50
            },
            "trades": []
        }
        
        variant_b_results = {
            "strategy_id": "test_strategy",
            "version_id": "v2.0",
            "metrics": {
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.15,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "trade_count": 60
            },
            "trades": []
        }
        
        # Compare results
        comparison = self.integration._compare_backtest_results(
            variant_a_results, variant_b_results
        )
        
        # Check comparison structure
        self.assertIn("b_is_better", comparison)
        self.assertIn("differences", comparison)
        self.assertIn("relative_improvements", comparison)
        self.assertIn("confidence_score", comparison)
        
        # Check that B is correctly identified as better
        self.assertTrue(comparison["b_is_better"])
        
        # Check differences
        differences = comparison["differences"]
        self.assertEqual(differences["sharpe_ratio"], 0.5)
        self.assertEqual(differences["max_drawdown"], 0.05)  # -0.15 - (-0.2)
        
        # Check confidence score is between 0 and 1
        self.assertGreaterEqual(comparison["confidence_score"], 0.0)
        self.assertLessEqual(comparison["confidence_score"], 1.0)
    
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
        
        # Check summary structure
        self.assertIn("total_regimes", summary)
        self.assertIn("regimes_b_better", summary)
        self.assertIn("promote_b", summary)
        self.assertIn("confidence", summary)
        self.assertIn("regime_specific", summary)
        
        # Check values
        self.assertEqual(summary["total_regimes"], 4)
        self.assertEqual(summary["regimes_b_better"], 3)
        self.assertTrue(summary["promote_b"])  # 3/4 > 0.75
        
        # Check regime-specific recommendations
        regime_specific = summary["regime_specific"]
        self.assertEqual(len(regime_specific), 4)
        self.assertEqual(regime_specific["bullish"]["recommended_variant"], "B")
        self.assertEqual(regime_specific["sideways"]["recommended_variant"], "A")


class TestSyntheticTestingEvents(unittest.TestCase):
    """Test event handling in the synthetic testing integration."""

    def setUp(self):
        """Set up test environment with mocks."""
        # Mock dependencies
        self.event_bus_mock = MagicMock()
        self.ab_test_manager_mock = MagicMock()
        self.approval_manager_mock = MagicMock()
        
        # Create patchers
        self.event_bus_patcher = patch('trading_bot.event_system.EventBus', 
                                      return_value=self.event_bus_mock)
        self.synthetic_generator_patcher = patch(
            'trading_bot.autonomous.synthetic_market_generator.SyntheticMarketGenerator'
        )
        self.correlated_generator_patcher = patch(
            'trading_bot.autonomous.synthetic_market_generator_correlations.CorrelatedMarketGenerator'
        )
        self.ab_test_manager_patcher = patch(
            'trading_bot.autonomous.synthetic_testing_integration.get_ab_test_manager',
            return_value=self.ab_test_manager_mock
        )
        self.ab_test_analyzer_patcher = patch(
            'trading_bot.autonomous.synthetic_testing_integration.get_ab_test_analyzer'
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
        
        # Create the integration instance with mock methods
        self.integration = SyntheticTestingIntegration()
        self.integration.add_synthetic_testing_to_test = MagicMock(return_value=True)
        self.integration.enhance_approval_request_with_synthetic_data = MagicMock(return_value=True)

    def tearDown(self):
        """Clean up test environment."""
        # Stop all patches
        self.event_bus_patcher.stop()
        self.synthetic_generator_patcher.stop()
        self.correlated_generator_patcher.stop()
        self.ab_test_manager_patcher.stop()
        self.ab_test_analyzer_patcher.stop()
        self.approval_manager_patcher.stop()

    def test_handle_test_created(self):
        """Test handling of test creation events."""
        # Create a mock test
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"use_synthetic_testing": True}
        
        # Configure manager to return our mock test
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create event with test data
        event = MagicMock()
        event.data = {"test_id": "test123"}
        
        # Call handler
        self.integration._handle_test_created(event)
        
        # Check that test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that synthetic testing was added
        self.integration.add_synthetic_testing_to_test.assert_called_once_with(mock_test)
    
    def test_handle_test_created_no_synthetic(self):
        """Test handling of test creation events when synthetic testing is disabled."""
        # Create a mock test with synthetic testing disabled
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"use_synthetic_testing": False}
        
        # Configure manager to return our mock test
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create event with test data
        event = MagicMock()
        event.data = {"test_id": "test123"}
        
        # Call handler
        self.integration._handle_test_created(event)
        
        # Check that test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that synthetic testing was not added
        self.integration.add_synthetic_testing_to_test.assert_not_called()
    
    def test_handle_approval_request_created(self):
        """Test handling of approval request creation events."""
        # Create a mock request and test
        mock_request = MagicMock()
        mock_request.request_id = "req123"
        mock_request.test_id = "test123"
        
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"synthetic_testing_completed": False}
        
        # Configure managers to return our mocks
        self.approval_manager_mock.get_request.return_value = mock_request
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create event with request data
        event = MagicMock()
        event.data = {
            "request_id": "req123",
            "test_id": "test123"
        }
        
        # Call handler
        self.integration._handle_approval_request_created(event)
        
        # Check that request was retrieved
        self.approval_manager_mock.get_request.assert_called_once_with("req123")
        
        # Check that test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that request was enhanced
        self.integration.enhance_approval_request_with_synthetic_data.assert_called_once_with(
            mock_request, mock_test
        )
    
    def test_handle_approval_request_already_processed(self):
        """Test handling when synthetic testing was already completed."""
        # Create a mock request and test with synthetic testing already completed
        mock_request = MagicMock()
        mock_request.request_id = "req123"
        mock_request.test_id = "test123"
        
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {"synthetic_testing_completed": True}
        
        # Configure managers to return our mocks
        self.approval_manager_mock.get_request.return_value = mock_request
        self.ab_test_manager_mock.get_test.return_value = mock_test
        
        # Create event with request data
        event = MagicMock()
        event.data = {
            "request_id": "req123",
            "test_id": "test123"
        }
        
        # Call handler
        self.integration._handle_approval_request_created(event)
        
        # Check that request was retrieved
        self.approval_manager_mock.get_request.assert_called_once_with("req123")
        
        # Check that test was retrieved
        self.ab_test_manager_mock.get_test.assert_called_once_with("test123")
        
        # Check that request was not enhanced
        self.integration.enhance_approval_request_with_synthetic_data.assert_not_called()


class TestSyntheticTestingSimulation(unittest.TestCase):
    """Test synthetic market simulation with different regimes."""

    def setUp(self):
        """Set up test environment with real components."""
        # Use real synthetic market generators for simulation tests
        self.synthetic_generator = SyntheticMarketGenerator()
        self.correlated_generator = CorrelatedMarketGenerator()
        
        # Create patchers for other components
        self.event_bus_mock = MagicMock()
        self.ab_test_manager_mock = MagicMock()
        self.ab_test_analyzer_mock = MagicMock()
        self.approval_manager_mock = MagicMock()
        
        self.event_bus_patcher = patch('trading_bot.event_system.EventBus', 
                                      return_value=self.event_bus_mock)
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
        
        # Start patchers for non-synthetic components
        self.event_bus_patcher.start()
        self.ab_test_manager_patcher.start()
        self.ab_test_analyzer_patcher.start()
        self.approval_manager_patcher.start()
        
        # Create integration with real synthetic generators
        with patch('trading_bot.autonomous.synthetic_market_generator.SyntheticMarketGenerator', 
                  return_value=self.synthetic_generator), \
             patch('trading_bot.autonomous.synthetic_market_generator_correlations.CorrelatedMarketGenerator',
                  return_value=self.correlated_generator):
            self.integration = SyntheticTestingIntegration()

    def tearDown(self):
        """Clean up test environment."""
        # Stop all patches
        self.event_bus_patcher.stop()
        self.ab_test_manager_patcher.stop()
        self.ab_test_analyzer_patcher.stop()
        self.approval_manager_patcher.stop()

    def test_generate_realistic_bullish_market(self):
        """Test generation of realistic bullish market data."""
        # Generate bullish market data
        market_data = self.integration._generate_regime_specific_data(
            symbols=["BTC", "ETH"],
            days=30,
            regime=MarketRegimeType.BULLISH
        )
        
        # Check that data was generated for all symbols
        self.assertIn("BTC", market_data)
        self.assertIn("ETH", market_data)
        
        # Check data characteristics
        for symbol, data in market_data.items():
            # Check dataframe structure
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 30)
            
            # Check price direction (should be generally up in bullish regime)
            first_price = data["price"].iloc[0]
            last_price = data["price"].iloc[-1]
            
            # In bullish markets, end price is usually higher than start price
            # This is a probabilistic test, but should be reliable with our configuration
            self.assertGreater(last_price, first_price)
    
    def test_generate_realistic_bearish_market(self):
        """Test generation of realistic bearish market data."""
        # Generate bearish market data
        market_data = self.integration._generate_regime_specific_data(
            symbols=["BTC", "ETH"],
            days=30,
            regime=MarketRegimeType.BEARISH
        )
        
        # Check data characteristics
        for symbol, data in market_data.items():
            # Check price direction (should be generally down in bearish regime)
            first_price = data["price"].iloc[0]
            last_price = data["price"].iloc[-1]
            
            # In bearish markets, end price is usually lower than start price
            # This is a probabilistic test, but should be reliable with our configuration
            self.assertLess(last_price, first_price)
    
    def test_generate_realistic_sideways_market(self):
        """Test generation of realistic sideways market data."""
        # Generate sideways market data
        market_data = self.integration._generate_regime_specific_data(
            symbols=["BTC", "ETH"],
            days=30,
            regime=MarketRegimeType.SIDEWAYS
        )
        
        # Check volatility (should be lower in sideways markets)
        for symbol, data in market_data.items():
            # Calculate volatility as standard deviation of returns
            returns = data["price"].pct_change().dropna()
            sideways_volatility = returns.std()
            
            # Generate volatile market for comparison
            volatile_data = self.integration._generate_regime_specific_data(
                symbols=[symbol],
                days=30,
                regime=MarketRegimeType.VOLATILE
            )[symbol]
            volatile_returns = volatile_data["price"].pct_change().dropna()
            volatile_volatility = volatile_returns.std()
            
            # Sideways volatility should be lower than volatile
            self.assertLess(sideways_volatility, volatile_volatility)
    
    def test_correlation_between_assets(self):
        """Test correlation between generated assets."""
        # Generate correlated market data
        market_data = self.integration._generate_regime_specific_data(
            symbols=["BTC", "ETH", "SOL", "ADA"],
            days=100,
            regime=MarketRegimeType.BULLISH
        )
        
        # Calculate returns for correlation analysis
        returns = {}
        for symbol, data in market_data.items():
            returns[symbol] = data["price"].pct_change().dropna()
        
        # Create returns dataframe
        returns_df = pd.DataFrame(returns)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Check that correlations are not too low or too high
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                # Correlation between different assets should be within realistic bounds
                correlation = corr_matrix.iloc[i, j]
                self.assertGreater(correlation, 0.1)  # Not too uncorrelated
                self.assertLess(correlation, 0.95)    # Not too perfectly correlated
    
    def test_end_to_end_simulation(self):
        """Test complete simulation workflow with all market regimes."""
        # Create a mock test
        mock_test = MagicMock()
        mock_test.test_id = "test123"
        mock_test.metadata = {}
        mock_test.variant_a = MagicMock()
        mock_test.variant_a.strategy_id = "strategy_a"
        mock_test.variant_a.version_id = "v1.0"
        mock_test.variant_b = MagicMock()
        mock_test.variant_b.strategy_id = "strategy_a"
        mock_test.variant_b.version_id = "v2.0"
        
        # Set up the add_synthetic_testing method to call the real implementation
        with patch.object(
            SyntheticTestingIntegration, 
            '_backtest_strategy', 
            return_value={
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.2,
                    "win_rate": 0.6,
                    "profit_factor": 1.5,
                    "trade_count": 50
                },
                "trades": []
            }
        ):
            # Add synthetic testing to the test
            self.integration.add_synthetic_testing_to_test(mock_test)
            
            # Check that metadata was updated
            self.assertTrue(mock_test.metadata.get("synthetic_testing_completed", False))
            self.assertIn("synthetic_testing_results", mock_test.metadata)
            self.assertIn("synthetic_testing_timestamp", mock_test.metadata)
            
            # Check that event was emitted
            self.event_bus_mock.emit.assert_called()
            
            # Check that test was updated in the manager
            self.ab_test_manager_mock.update_test.assert_called_with(mock_test)
            
            # Check results for each regime
            results = mock_test.metadata["synthetic_testing_results"]
            for regime in self.integration.default_regimes:
                regime_value = regime.value
                self.assertIn(regime_value, results)
                self.assertIn("variant_a", results[regime_value])
                self.assertIn("variant_b", results[regime_value])
                self.assertIn("comparison", results[regime_value])


if __name__ == "__main__":
    unittest.main()
