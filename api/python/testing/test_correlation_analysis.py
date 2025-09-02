#!/usr/bin/env python3
"""
Test Correlation Analysis

This module tests the correlation analysis components of the autonomous trading system:
- CorrelationMatrix for calculating and tracking correlations
- CorrelationMonitor for adjusting allocations based on correlations

Tests use synthetic data to simulate strategy returns and verify behavior.
"""

import os
import sys
import unittest
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import threading
import time

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components to test
from trading_bot.autonomous.correlation_matrix import CorrelationMatrix
from trading_bot.autonomous.correlation_monitor import CorrelationMonitor, get_correlation_monitor
from trading_bot.event_system import EventBus, Event, EventType
from trading_bot.testing.market_data_generator import MarketDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestCorrelationMatrix(unittest.TestCase):
    """Test CorrelationMatrix functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.matrix = CorrelationMatrix(
            window_size=20,
            min_periods=5,
            correlation_method='pearson'
        )
        
        # Create test directory
        self.test_dir = os.path.join(
            os.path.dirname(__file__),
            'test_data',
            'correlation'
        )
        os.makedirs(self.test_dir, exist_ok=True)
    
    def test_add_return_data(self):
        """Test adding return data to the matrix."""
        # Add data for two strategies
        self.matrix.add_return_data('strategy1', datetime(2023, 1, 1), 0.01)
        self.matrix.add_return_data('strategy1', datetime(2023, 1, 2), 0.02)
        self.matrix.add_return_data('strategy2', datetime(2023, 1, 1), -0.01)
        self.matrix.add_return_data('strategy2', datetime(2023, 1, 2), 0.03)
        
        # Verify that data was added correctly
        self.assertEqual(self.matrix.returns_data.shape, (2, 2))  # 2 days, 2 strategies
        self.assertEqual(self.matrix.returns_data.loc[pd.Timestamp('2023-01-01'), 'strategy1'], 0.01)
        self.assertEqual(self.matrix.returns_data.loc[pd.Timestamp('2023-01-02'), 'strategy2'], 0.03)
    
    def test_calculate_correlation(self):
        """Test calculating correlations."""
        # Add data with perfect positive correlation
        for i in range(10):
            day = datetime(2023, 1, 1) + timedelta(days=i)
            value = 0.01 * (i+1)
            self.matrix.add_return_data('strategy1', day, value)
            self.matrix.add_return_data('strategy2', day, value)
        
        # Calculate correlation
        correlation = self.matrix.calculate_correlation()
        
        # Should be perfect correlation (1.0)
        self.assertAlmostEqual(correlation.loc['strategy1', 'strategy2'], 1.0, places=5)
        
        # Add data with perfect negative correlation
        for i in range(10):
            day = datetime(2023, 1, 11) + timedelta(days=i)
            value = 0.01 * (i+1)
            self.matrix.add_return_data('strategy3', day, value)
            self.matrix.add_return_data('strategy4', day, -value)
        
        # Calculate correlation for all strategies
        correlation = self.matrix.calculate_correlation()
        
        # Check expected correlations
        self.assertAlmostEqual(correlation.loc['strategy1', 'strategy2'], 1.0, places=5)
        self.assertAlmostEqual(correlation.loc['strategy3', 'strategy4'], -1.0, places=5)
        self.assertTrue(abs(correlation.loc['strategy1', 'strategy3']) < 1.0)  # Should be less correlated
    
    def test_significant_changes(self):
        """Test detection of significant correlation changes."""
        # Add initial data
        for i in range(10):
            day = datetime(2023, 1, 1) + timedelta(days=i)
            value1 = 0.01 * (i+1)
            value2 = 0.01 * (i+1)  # Perfectly correlated initially
            self.matrix.add_return_data('strategy1', day, value1)
            self.matrix.add_return_data('strategy2', day, value2)
        
        # Calculate initial correlation
        self.matrix.calculate_correlation()
        
        # Now add data with negative correlation to cause a significant change
        for i in range(10):
            day = datetime(2023, 1, 11) + timedelta(days=i)
            value1 = 0.01 * (i+1)
            value2 = -0.01 * (i+1)  # Negative correlation
            self.matrix.add_return_data('strategy1', day, value1)
            self.matrix.add_return_data('strategy2', day, value2)
        
        # Calculate correlation again
        self.matrix.calculate_correlation()
        
        # Should have detected a significant change
        self.assertTrue(len(self.matrix.significant_changes) > 0)
        
        # Verify change details
        if self.matrix.significant_changes:
            pair, old_val, new_val, _ = self.matrix.significant_changes[0]
            self.assertEqual(set(pair), {'strategy1', 'strategy2'})
            self.assertTrue(old_val > 0.5)  # Old value should be positive
            self.assertTrue(new_val < 0)    # New value should be negative
    
    def test_highly_correlated_pairs(self):
        """Test finding highly correlated pairs."""
        # Create three strategies with varying correlation
        for i in range(20):
            day = datetime(2023, 1, 1) + timedelta(days=i)
            
            # strategy1 and strategy2 are highly positively correlated
            value1 = 0.01 * (i+1)
            value2 = 0.01 * (i+1) + 0.001  # Almost identical
            
            # strategy3 is less correlated with the others
            value3 = 0.01 * np.sin(i)
            
            self.matrix.add_return_data('strategy1', day, value1)
            self.matrix.add_return_data('strategy2', day, value2)
            self.matrix.add_return_data('strategy3', day, value3)
        
        # Calculate correlation
        self.matrix.calculate_correlation()
        
        # Get highly correlated pairs with threshold 0.8
        pairs = self.matrix.get_highly_correlated_pairs(threshold=0.8)
        
        # Should find one highly correlated pair
        self.assertEqual(len(pairs), 1)
        if pairs:
            s1, s2, corr = pairs[0]
            self.assertEqual(set([s1, s2]), {'strategy1', 'strategy2'})
            self.assertTrue(corr > 0.8)
    
    def test_serialization(self):
        """Test serialization to and from dict."""
        # Add data for two strategies
        for i in range(10):
            day = datetime(2023, 1, 1) + timedelta(days=i)
            self.matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.matrix.add_return_data('strategy2', day, -0.01 * (i+1))
        
        # Calculate correlation
        original_corr = self.matrix.calculate_correlation()
        
        # Serialize to dict
        data_dict = self.matrix.to_dict()
        
        # Create new matrix from dict
        new_matrix = CorrelationMatrix.from_dict(data_dict)
        
        # Calculate correlation with new matrix
        new_corr = new_matrix.calculate_correlation()
        
        # Should be the same
        self.assertEqual(
            original_corr.loc['strategy1', 'strategy2'],
            new_corr.loc['strategy1', 'strategy2']
        )
    
    def test_persistence(self):
        """Test saving to and loading from file."""
        # Add data
        for i in range(10):
            day = datetime(2023, 1, 1) + timedelta(days=i)
            self.matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.matrix.add_return_data('strategy2', day, -0.01 * (i+1))
        
        # Calculate correlation
        original_corr = self.matrix.calculate_correlation()
        
        # Save to file
        filepath = os.path.join(self.test_dir, 'corr_matrix_test.json')
        self.matrix.save_to_file(filepath)
        
        # Load from file
        loaded_matrix = CorrelationMatrix.load_from_file(filepath)
        
        # Calculate correlation with loaded matrix
        loaded_corr = loaded_matrix.calculate_correlation()
        
        # Should be the same
        self.assertEqual(
            original_corr.loc['strategy1', 'strategy2'],
            loaded_corr.loc['strategy1', 'strategy2']
        )


class TestCorrelationMonitor(unittest.TestCase):
    """Test CorrelationMonitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create event bus
        self.event_bus = EventBus()
        
        # Create test directory
        self.test_dir = os.path.join(
            os.path.dirname(__file__),
            'test_data',
            'correlation_monitor'
        )
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create monitor with patched risk manager and deployment pipeline
        with patch('trading_bot.autonomous.correlation_monitor.get_autonomous_risk_manager') as mock_get_risk:
            with patch('trading_bot.autonomous.correlation_monitor.get_deployment_pipeline') as mock_get_pipeline:
                # Mock risk manager
                self.mock_risk_manager = MagicMock()
                mock_get_risk.return_value = self.mock_risk_manager
                
                # Mock deployment pipeline
                self.mock_pipeline = MagicMock()
                mock_get_pipeline.return_value = self.mock_pipeline
                
                # Configure mock pipeline
                self.mock_pipeline.get_deployments.return_value = [
                    {
                        "deployment_id": "deploy1",
                        "strategy_id": "strategy1",
                        "status": "ACTIVE",
                        "risk_params": {"allocation_percentage": 10.0}
                    },
                    {
                        "deployment_id": "deploy2",
                        "strategy_id": "strategy2",
                        "status": "ACTIVE",
                        "risk_params": {"allocation_percentage": 15.0}
                    },
                    {
                        "deployment_id": "deploy3",
                        "strategy_id": "strategy3",
                        "status": "ACTIVE",
                        "risk_params": {"allocation_percentage": 8.0}
                    }
                ]
                
                # Create monitor
                self.monitor = CorrelationMonitor(
                    event_bus=self.event_bus,
                    persistence_dir=self.test_dir
                )
        
        # Track emitted events
        self.emitted_events = []
        self.event_bus.register("*", self._track_events)
    
    def _track_events(self, event_type, data):
        """Track events for verification."""
        self.emitted_events.append((event_type, data))
    
    def test_event_handling(self):
        """Test event handling for return data collection."""
        # Emit a trade execution event
        self.event_bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            source="TestSource",
            data={
                "strategy_id": "strategy1",
                "timestamp": datetime.now(),
                "profit_loss": 100.0
            }
        ))
        
        # Emit a position closed event
        self.event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            source="TestSource",
            data={
                "strategy_id": "strategy2",
                "timestamp": datetime.now(),
                "profit_loss": 200.0,
                "profit_loss_pct": 2.5
            }
        ))
        
        # Wait for event processing
        time.sleep(0.1)
        
        # Should have added return data for both strategies
        self.assertIn('strategy1', self.monitor.correlation_matrix.returns_data.columns)
        self.assertIn('strategy2', self.monitor.correlation_matrix.returns_data.columns)
    
    def test_correlation_threshold_detection(self):
        """Test detection of strategies exceeding correlation threshold."""
        # Add highly correlated return data
        for i in range(20):
            day = datetime.now() - timedelta(days=20-i)
            # Almost perfect correlation
            self.monitor.correlation_matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy2', day, 0.01 * (i+1) + 0.0001)
        
        # Run correlation check
        self.monitor._check_correlation_thresholds()
        
        # Should have emitted a threshold exceeded event
        threshold_events = [e for e in self.emitted_events if e[0] == "CORRELATION_THRESHOLD_EXCEEDED"]
        self.assertTrue(len(threshold_events) > 0)
        
        if threshold_events:
            event_type, data = threshold_events[0]
            self.assertEqual(set([data['strategy1'], data['strategy2']]), {'strategy1', 'strategy2'})
            self.assertTrue(data['correlation'] > self.monitor.config['high_correlation_threshold'])
    
    def test_allocation_adjustment(self):
        """Test allocation adjustment based on correlation."""
        # Add highly correlated return data
        for i in range(20):
            day = datetime.now() - timedelta(days=20-i)
            # Almost perfect correlation
            self.monitor.correlation_matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy2', day, 0.01 * (i+1) + 0.0001)
        
        # Add less correlated data for strategy3
        for i in range(20):
            day = datetime.now() - timedelta(days=20-i)
            self.monitor.correlation_matrix.add_return_data('strategy3', day, -0.01 * np.sin(i))
        
        # Setup mock for risk manager's adjust_allocation method
        self.mock_risk_manager.adjust_allocation.return_value = True
        
        # Run allocation adjustment
        self.monitor._adjust_allocations_for_correlation()
        
        # Verify risk manager was called to adjust allocation
        self.mock_risk_manager.adjust_allocation.assert_called()
        
        # Should have adjusted the strategy with higher allocation
        call_args = self.mock_risk_manager.adjust_allocation.call_args_list
        adjusted_strategy = None
        for args, kwargs in call_args:
            strategy_id, new_allocation = args
            if strategy_id in ['strategy1', 'strategy2']:
                adjusted_strategy = strategy_id
                self.assertTrue(new_allocation < 15.0)  # Should be reduced
        
        self.assertIsNotNone(adjusted_strategy)
        
        # Should have emitted an allocation adjusted event
        allocation_events = [e for e in self.emitted_events if e[0] == "ALLOCATION_ADJUSTED_FOR_CORRELATION"]
        self.assertTrue(len(allocation_events) > 0)
    
    def test_correlation_report_generation(self):
        """Test generation of correlation report."""
        # Add return data
        for i in range(20):
            day = datetime.now() - timedelta(days=20-i)
            self.monitor.correlation_matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy2', day, -0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy3', day, 0.01 * np.sin(i))
        
        # Generate report
        self.monitor._generate_correlation_report()
        
        # Should have emitted a report event
        report_events = [e for e in self.emitted_events if e[0] == "CORRELATION_REPORT_GENERATED"]
        self.assertTrue(len(report_events) > 0)
        
        if report_events:
            event_type, data = report_events[0]
            self.assertIn('statistics', data)
            self.assertIn('highly_correlated_pairs', data)
            
            # Check statistics
            stats = data['statistics']
            self.assertEqual(stats['tracked_strategies'], 3)
    
    def test_heatmap_data_generation(self):
        """Test generation of heatmap visualization data."""
        # Add return data
        for i in range(20):
            day = datetime.now() - timedelta(days=20-i)
            self.monitor.correlation_matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy2', day, -0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy3', day, 0.01 * np.sin(i))
        
        # Get heatmap data
        heatmap_data = self.monitor.get_correlation_heatmap_data()
        
        # Check structure
        self.assertIn('strategies', heatmap_data)
        self.assertIn('matrix', heatmap_data)
        
        # Should have 3 strategies
        self.assertEqual(len(heatmap_data['strategies']), 3)
        
        # Matrix should be 3x3
        self.assertEqual(len(heatmap_data['matrix']), 3)
        self.assertEqual(len(heatmap_data['matrix'][0]), 3)
        
        # Diagonal should be 1.0 (self-correlation)
        for i in range(3):
            self.assertAlmostEqual(heatmap_data['matrix'][i][i], 1.0)
    
    def test_persistence(self):
        """Test persistence of monitor state."""
        # Add return data
        for i in range(20):
            day = datetime.now() - timedelta(days=20-i)
            self.monitor.correlation_matrix.add_return_data('strategy1', day, 0.01 * (i+1))
            self.monitor.correlation_matrix.add_return_data('strategy2', day, -0.01 * (i+1))
        
        # Record an allocation adjustment
        self.monitor.allocation_adjustments['strategy1'] = {
            'timestamp': datetime.now().isoformat(),
            'original_allocation': 10.0,
            'new_allocation': 8.0,
            'adjustment_factor': 0.8,
            'correlations': [{'other_strategy': 'strategy2', 'correlation': 0.9}]
        }
        
        # Save state
        self.monitor._save_state()
        
        # Create a new monitor instance
        with patch('trading_bot.autonomous.correlation_monitor.get_autonomous_risk_manager') as mock_get_risk:
            with patch('trading_bot.autonomous.correlation_monitor.get_deployment_pipeline') as mock_get_pipeline:
                mock_get_risk.return_value = MagicMock()
                mock_get_pipeline.return_value = MagicMock()
                
                # Create new monitor with same persistence dir
                new_monitor = CorrelationMonitor(
                    event_bus=self.event_bus,
                    persistence_dir=self.test_dir
                )
        
        # Verify state was loaded
        self.assertIn('strategy1', new_monitor.allocation_adjustments)
        self.assertEqual(
            new_monitor.allocation_adjustments['strategy1']['original_allocation'], 
            10.0
        )
        
        # Verify correlation data was loaded
        self.assertIn('strategy1', new_monitor.correlation_matrix.returns_data.columns)
        self.assertIn('strategy2', new_monitor.correlation_matrix.returns_data.columns)
    
    def test_singleton_pattern(self):
        """Test singleton pattern for global access."""
        # Reset the singleton instance
        with patch('trading_bot.autonomous.correlation_monitor._correlation_monitor', None):
            # Get instance
            monitor1 = get_correlation_monitor(self.event_bus)
            monitor2 = get_correlation_monitor(self.event_bus)
            
            # Should be the same instance
            self.assertIs(monitor1, monitor2)


def generate_synthetic_data():
    """Generate synthetic data for manual testing."""
    # Create market data generator
    generator = MarketDataGenerator(
        base_equity=10000,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),
        tickers=['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT'],
        price_paths=3,
        volatility=0.2
    )
    
    # Generate market data
    market_data = generator.generate_price_history()
    
    # Extract daily returns
    returns = {}
    for ticker in market_data:
        if ticker == 'date':
            continue
            
        # Calculate daily returns
        ticker_data = market_data[ticker]
        returns[ticker] = []
        for i in range(1, len(ticker_data)):
            prev_price = ticker_data[i-1]
            curr_price = ticker_data[i]
            daily_return = (curr_price - prev_price) / prev_price * 100
            returns[ticker].append(daily_return)
    
    # Create a correlation matrix
    matrix = CorrelationMatrix(window_size=20, min_periods=5)
    
    # Add return data
    for ticker in returns:
        for i, ret in enumerate(returns[ticker]):
            day = datetime(2023, 1, 1) + timedelta(days=i+1)
            matrix.add_return_data(ticker, day, ret)
    
    # Calculate and print correlation matrix
    correlation = matrix.calculate_correlation()
    print("\nSynthetic Data Correlation Matrix:")
    print(correlation)
    
    # Print highly correlated pairs
    pairs = matrix.get_highly_correlated_pairs(threshold=0.7)
    print("\nHighly Correlated Pairs (>0.7):")
    for s1, s2, corr in pairs:
        print(f"  {s1} <-> {s2}: {corr:.4f}")
    
    return matrix


def run_tests():
    """Run all tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


def manual_test():
    """Run manual tests with synthetic data."""
    print("\n=== Running Manual Tests with Synthetic Data ===")
    
    # Generate synthetic data
    matrix = generate_synthetic_data()
    
    # Create a monitor with mock components
    event_bus = EventBus()
    
    with patch('trading_bot.autonomous.correlation_monitor.get_autonomous_risk_manager') as mock_get_risk:
        with patch('trading_bot.autonomous.correlation_monitor.get_deployment_pipeline') as mock_get_pipeline:
            # Mock risk manager
            mock_risk_manager = MagicMock()
            mock_get_risk.return_value = mock_risk_manager
            
            # Mock deployment pipeline
            mock_pipeline = MagicMock()
            mock_get_pipeline.return_value = mock_pipeline
            
            # Configure mock pipeline
            mock_pipeline.get_deployments.return_value = [
                {
                    "deployment_id": f"deploy-{ticker}",
                    "strategy_id": ticker,
                    "status": "ACTIVE",
                    "risk_params": {"allocation_percentage": 10.0}
                }
                for ticker in ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
            ]
            
            # Create monitor
            monitor = CorrelationMonitor(event_bus=event_bus)
            
            # Set correlation matrix
            monitor.correlation_matrix = matrix
            
            # Check correlations
            monitor._check_correlation_thresholds()
            
            # Adjust allocations
            monitor._adjust_allocations_for_correlation()
            
            # Print allocation adjustments
            print("\nAllocation Adjustments:")
            for strategy_id, adjustment in monitor.allocation_adjustments.items():
                print(f"  {strategy_id}: {adjustment['original_allocation']:.1f}% -> {adjustment['new_allocation']:.1f}%")
            
            # Generate report
            report = monitor.get_correlation_report()
            print(f"\nTracking {report['statistics']['tracked_strategies']} strategies with average correlation: {report['statistics']['avg_correlation']:.4f}")
            
            # Get heatmap data
            heatmap = monitor.get_correlation_heatmap_data()
            print("\nHeatmap Data Structure:")
            print(f"  Strategies: {heatmap['strategies']}")
            print(f"  Matrix Shape: {len(heatmap['matrix'])}x{len(heatmap['matrix'][0])}")


if __name__ == '__main__':
    # Determine run mode
    if len(sys.argv) > 1 and sys.argv[1] == '--manual':
        manual_test()
    else:
        run_tests()
