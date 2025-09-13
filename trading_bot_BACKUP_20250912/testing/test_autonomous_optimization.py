#!/usr/bin/env python3
"""
Unit Tests for Autonomous Engine Optimization

This module tests the near-miss candidate detection and optimization
functionality in the autonomous engine, ensuring it properly identifies
strategies that can be improved.
"""

import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components to test
from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.event_system import EventBus, EventType

class NearMissDetectionTests(unittest.TestCase):
    """Tests for near-miss candidate detection and optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create autonomous engine with a mock event bus
        self.event_bus = EventBus()
        self.engine = AutonomousEngine()
        
        # Test thresholds
        self.thresholds = {
            "min_sharpe_ratio": 1.5,
            "min_profit_factor": 1.8,
            "max_drawdown": 15.0,
            "min_win_rate": 55.0
        }
        
        # Mock event handler to capture optimization events
        self.optimization_events = []
        self.event_bus.register(EventType.STRATEGY_OPTIMISED, self.capture_optimization_event)
        self.event_bus.register(EventType.STRATEGY_EXHAUSTED, self.capture_optimization_event)
    
    def capture_optimization_event(self, event_type, event_data):
        """Capture optimization events for testing."""
        self.optimization_events.append({
            "type": event_type,
            "data": event_data
        })
    
    def test_meets_performance_thresholds(self):
        """Test that _meets_performance_thresholds correctly evaluates metrics."""
        # Test cases with metrics and expected results
        test_cases = [
            # All metrics meet thresholds
            ({
                "sharpe_ratio": 1.8,
                "profit_factor": 2.0,
                "max_drawdown": 10.0,
                "win_rate": 60.0
            }, True),
            
            # One metric fails
            ({
                "sharpe_ratio": 1.4,  # Below threshold
                "profit_factor": 2.0,
                "max_drawdown": 10.0,
                "win_rate": 60.0
            }, False),
            
            # Multiple metrics fail
            ({
                "sharpe_ratio": 1.4,  # Below
                "profit_factor": 1.7,  # Below
                "max_drawdown": 16.0,  # Worse than max
                "win_rate": 52.0       # Below
            }, False),
            
            # Edge case - barely meets thresholds
            ({
                "sharpe_ratio": 1.5,    # Exactly at threshold
                "profit_factor": 1.8,   # Exactly at threshold
                "max_drawdown": 15.0,   # Exactly at threshold
                "win_rate": 55.0        # Exactly at threshold
            }, True),
            
            # Missing metrics
            ({
                "sharpe_ratio": 1.8,
                "profit_factor": 2.0
                # Missing max_drawdown and win_rate
            }, False)
        ]
        
        # Test each case
        for metrics, expected_result in test_cases:
            result = self.engine._meets_performance_thresholds(metrics, self.thresholds)
            self.assertEqual(result, expected_result, 
                             f"Failed for metrics: {metrics}, expected: {expected_result}")
    
    def test_is_near_miss_candidate(self):
        """Test that _is_near_miss_candidate correctly identifies candidates."""
        # Define test cases with different performance metrics
        test_cases = [
            # Not near-miss (too far below thresholds)
            ({
                "sharpe_ratio": 1.0,    # Well below threshold
                "profit_factor": 1.5,   # Well below threshold
                "max_drawdown": 20.0,   # Well above max
                "win_rate": 45.0        # Well below threshold
            }, False),
            
            # Near-miss (all metrics within 15% of thresholds)
            ({
                "sharpe_ratio": 1.3,    # 13% below threshold
                "profit_factor": 1.6,   # 11% below threshold
                "max_drawdown": 17.0,   # 13% worse than max
                "win_rate": 50.0        # 9% below threshold
            }, True),
            
            # Near-miss (some metrics already meet thresholds)
            ({
                "sharpe_ratio": 1.6,    # Above threshold
                "profit_factor": 1.6,   # 11% below threshold
                "max_drawdown": 14.0,   # Better than max
                "win_rate": 50.0        # 9% below threshold
            }, True),
            
            # Not near-miss (one metric too far off)
            ({
                "sharpe_ratio": 1.3,    # 13% below threshold
                "profit_factor": 1.6,   # 11% below threshold
                "max_drawdown": 20.0,   # 33% worse than max (too far)
                "win_rate": 50.0        # 9% below threshold
            }, False),
            
            # Edge case - exactly at near-miss threshold (assuming 85%)
            ({
                "sharpe_ratio": 1.275,  # 15% below
                "profit_factor": 1.53,  # 15% below
                "max_drawdown": 17.25,  # 15% worse
                "win_rate": 46.75       # 15% below
            }, True)
        ]
        
        # Test each case
        for metrics, expected_result in test_cases:
            result = self.engine._is_near_miss_candidate(metrics, self.thresholds)
            self.assertEqual(result, expected_result, 
                             f"Failed for metrics: {metrics}, expected: {expected_result}")
    
    def test_optimize_near_miss_candidates(self):
        """Test the optimization of near-miss candidates."""
        # Create mock strategy candidates
        candidates = []
        
        # Top performer (already meets thresholds)
        top_performer = MagicMock()
        top_performer.strategy_id = "top_strategy"
        top_performer.strategy_type = "Iron Condor"
        top_performer.performance_metrics = {
            "sharpe_ratio": 1.8,
            "profit_factor": 2.0,
            "max_drawdown": 10.0,
            "win_rate": 60.0
        }
        top_performer.status = "evaluated"
        candidates.append(top_performer)
        
        # Near-miss candidate
        near_miss = MagicMock()
        near_miss.strategy_id = "near_miss_strategy"
        near_miss.strategy_type = "Strangle"
        near_miss.performance_metrics = {
            "sharpe_ratio": 1.3,
            "profit_factor": 1.6,
            "max_drawdown": 17.0,
            "win_rate": 50.0
        }
        near_miss.status = "evaluated"
        candidates.append(near_miss)
        
        # Poor performer (not worth optimizing)
        poor_performer = MagicMock()
        poor_performer.strategy_id = "poor_strategy"
        poor_performer.strategy_type = "Butterfly"
        poor_performer.performance_metrics = {
            "sharpe_ratio": 0.9,
            "profit_factor": 1.2,
            "max_drawdown": 25.0,
            "win_rate": 40.0
        }
        poor_performer.status = "evaluated"
        candidates.append(poor_performer)
        
        # Mock the optimizer
        mock_optimizer = MagicMock()
        
        # Configure mock optimizer to "improve" the near-miss candidate
        def mock_optimize(candidate):
            if candidate.strategy_id == "near_miss_strategy":
                # Simulate improvement
                candidate.performance_metrics = {
                    "sharpe_ratio": 1.6,  # Improved
                    "profit_factor": 1.9,  # Improved
                    "max_drawdown": 14.0,  # Improved
                    "win_rate": 58.0       # Improved
                }
                candidate.status = "optimized"
                return True
            elif candidate.strategy_id == "poor_strategy":
                # Simulate failed optimization
                candidate.status = "optimization_exhausted"
                return False
            return False
            
        mock_optimizer.optimize = mock_optimize
        
        # Mock the emit_event method
        self.engine._emit_event = MagicMock()
        
        # Set the optimizer
        self.engine.optimizer = mock_optimizer
        
        # Run optimization
        self.engine._optimize_near_miss_candidates(candidates, self.thresholds)
        
        # Verify only near-miss candidate was optimized
        mock_optimizer.optimize.assert_called_once()
        
        # Verify the correct candidate was optimized
        self.assertEqual(near_miss.status, "optimized")
        self.assertEqual(top_performer.status, "evaluated")  # Unchanged
        self.assertEqual(poor_performer.status, "evaluated")  # Unchanged
        
        # Verify events were emitted
        self.engine._emit_event.assert_called_once()
        
        # Verify the event was for the near-miss candidate
        args, kwargs = self.engine._emit_event.call_args
        self.assertEqual(args[0], EventType.STRATEGY_OPTIMISED)
        self.assertEqual(args[1]["strategy_id"], "near_miss_strategy")
    
    def test_integration_with_event_system(self):
        """Test the integration with the event system."""
        # Replace the actual emit_event method to use our event bus
        def mock_emit_event(event_type, event_data):
            self.event_bus.emit(event_type, event_data)
            
        self.engine._emit_event = mock_emit_event
        
        # Create a mock optimizer that always succeeds
        mock_optimizer = MagicMock()
        def always_improve(candidate):
            # Simulate improvement
            candidate.performance_metrics = {
                "sharpe_ratio": 1.8,  # Improved
                "profit_factor": 2.0,  # Improved
                "max_drawdown": 12.0,  # Improved
                "win_rate": 60.0       # Improved
            }
            candidate.status = "optimized"
            return True
            
        mock_optimizer.optimize = always_improve
        self.engine.optimizer = mock_optimizer
        
        # Create a near-miss candidate
        near_miss = MagicMock()
        near_miss.strategy_id = "event_test_strategy"
        near_miss.strategy_type = "Strangle"
        near_miss.performance_metrics = {
            "sharpe_ratio": 1.3,
            "profit_factor": 1.6,
            "max_drawdown": 17.0,
            "win_rate": 50.0
        }
        near_miss.status = "evaluated"
        
        # Optimize it
        candidates = [near_miss]
        self.engine._optimize_near_miss_candidates(candidates, self.thresholds)
        
        # Verify an event was captured
        self.assertEqual(len(self.optimization_events), 1)
        
        # Verify event data
        event = self.optimization_events[0]
        self.assertEqual(event["type"], EventType.STRATEGY_OPTIMISED)
        self.assertEqual(event["data"]["strategy_id"], "event_test_strategy")
        self.assertEqual(event["data"]["strategy_type"], "Strangle")
        
        # Verify before/after metrics are included
        self.assertIn("before_metrics", event["data"])
        self.assertIn("after_metrics", event["data"])
        
        # Verify improvement
        before = event["data"]["before_metrics"]["sharpe_ratio"]
        after = event["data"]["after_metrics"]["sharpe_ratio"]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
