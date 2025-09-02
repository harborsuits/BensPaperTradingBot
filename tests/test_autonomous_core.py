#!/usr/bin/env python3
"""
Simplified Core Test for Autonomous Trading Engine

This script tests the core functionality of the autonomous trading engine
with mocked market data, focusing on:
1. Strategy adapter integration
2. Near-miss candidate identification 
3. Event emission validation

No external dependencies like yfinance required.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_core_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("autonomous_test")

# Import core components only
from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.event_system import EventBus, EventType
from trading_bot.strategies.components.component_registry import ComponentRegistry
from trading_bot.strategies.strategy_adapter import StrategyAdapter

# Simple event listener for testing
class CoreEventListener:
    def __init__(self):
        self.event_log = []
        self.register_events()
        
    def register_events(self):
        event_bus = EventBus()
        event_bus.register(EventType.STRATEGY_OPTIMISED, self.log_event)
        event_bus.register(EventType.STRATEGY_EXHAUSTED, self.log_event)
        event_bus.register(EventType.STRATEGY_GENERATED, self.log_event)
        event_bus.register(EventType.STRATEGY_EVALUATED, self.log_event)
        
    def log_event(self, event_type, event_data):
        logger.info(f"Event received: {event_type} with data: {event_data}")
        self.event_log.append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.datetime.now()
        })
        
    def get_events_by_type(self, event_type):
        return [e for e in self.event_log if e["type"] == event_type]
        
    def get_event_summary(self):
        summary = {}
        for event in self.event_log:
            event_type = event["type"]
            if event_type in summary:
                summary[event_type] += 1
            else:
                summary[event_type] = 1
        return summary


class MockedStrategyTest:
    """Test with mocked strategies and market data"""
    
    def __init__(self):
        self.engine = AutonomousEngine()
        self.event_listener = CoreEventListener()
        self.component_registry = ComponentRegistry()
        
        # Prepare for tests
        self._patch_engine_for_testing()
        
    def _patch_engine_for_testing(self):
        """Patch the engine to bypass actual market data and backtesting"""
        
        # Store original methods to restore later
        self.original_scan_market = self.engine._scan_market
        self.original_backtest = self.engine._backtest_candidate
        
        # Replace with test versions
        def mock_scan_market(universe, strategy_type):
            """Mock market scanning to return test symbols"""
            logger.info(f"Mock scanning {universe} for {strategy_type}")
            return ["AAPL", "MSFT", "GOOG", "AMZN"]
            
        def mock_backtest(candidate):
            """Mock backtesting to return predefined performance metrics"""
            logger.info(f"Mock backtesting {candidate.strategy_id}")
            
            # Create different performance profiles based on strategy type
            # to test the near-miss identification logic
            strategy_type = candidate.strategy_type
            
            # Each strategy type gets different performance metrics to test
            # both passing, near-miss, and failing scenarios
            if strategy_type == "Iron Condor":
                if "AAPL" in candidate.symbols:
                    # Top performer
                    candidate.performance_metrics = {
                        "sharpe_ratio": 1.8,
                        "profit_factor": 2.0,
                        "max_drawdown": 10.0,
                        "win_rate": 60.0
                    }
                elif "MSFT" in candidate.symbols:
                    # Near-miss candidate (85% of thresholds)
                    candidate.performance_metrics = {
                        "sharpe_ratio": 1.3,  # Below min of 1.5
                        "profit_factor": 1.7,  # Below min of 1.8
                        "max_drawdown": 16.0,  # Worse than max of 15.0
                        "win_rate": 52.0       # Below min of 55.0
                    }
                else:
                    # Poor performer
                    candidate.performance_metrics = {
                        "sharpe_ratio": 0.9,
                        "profit_factor": 1.3,
                        "max_drawdown": 22.0,
                        "win_rate": 45.0
                    }
                    
            elif strategy_type == "Strangle":
                if "GOOG" in candidate.symbols:
                    # Top performer
                    candidate.performance_metrics = {
                        "sharpe_ratio": 1.9,
                        "profit_factor": 2.2,
                        "max_drawdown": 8.0,
                        "win_rate": 65.0
                    }
                else:
                    # Near-miss candidate
                    candidate.performance_metrics = {
                        "sharpe_ratio": 1.4,
                        "profit_factor": 1.6,
                        "max_drawdown": 17.0,
                        "win_rate": 51.0
                    }
                    
            elif strategy_type == "Butterfly Spread":
                # All near-miss
                candidate.performance_metrics = {
                    "sharpe_ratio": 1.45,
                    "profit_factor": 1.75,
                    "max_drawdown": 16.5,
                    "win_rate": 53.0
                }
                
            else:
                # Default - poor performance
                candidate.performance_metrics = {
                    "sharpe_ratio": 1.0,
                    "profit_factor": 1.2,
                    "max_drawdown": 20.0,
                    "win_rate": 48.0
                }
                
            candidate.status = "backtested"
            return True
            
        # Apply patches
        self.engine._scan_market = mock_scan_market
        self.engine._backtest_candidate = mock_backtest
        
    def restore_engine_methods(self):
        """Restore original engine methods"""
        self.engine._scan_market = self.original_scan_market
        self.engine._backtest_candidate = self.original_backtest
        
    def test_strategy_adapters(self):
        """Test that all strategies are correctly adapted"""
        logger.info("Testing strategy adapters")
        
        # Get all registered strategy types
        strategies = self.component_registry.get_registered_strategy_types()
        logger.info(f"Found registered strategies: {strategies}")
        
        if not strategies:
            logger.error("No strategies registered in component registry")
            return False
            
        # Test instantiation and adapter for each strategy type
        success = True
        for strategy_type in strategies:
            try:
                strategy = self.component_registry.get_strategy_instance(strategy_type)
                
                # Verify adapter or interface compliance
                if isinstance(strategy, StrategyAdapter):
                    logger.info(f"Strategy {strategy_type} correctly wrapped with adapter")
                elif hasattr(strategy, 'generate_signals') and hasattr(strategy, 'size_position'):
                    logger.info(f"Strategy {strategy_type} natively implements the required interface")
                else:
                    logger.error(f"Strategy {strategy_type} not properly adapted")
                    success = False
                    
                # Verify core methods existence
                methods_to_check = ['generate_signals', 'size_position', 'manage_open_trades']
                for method in methods_to_check:
                    if not hasattr(strategy, method):
                        logger.error(f"Strategy {strategy_type} missing required method: {method}")
                        success = False
                
            except Exception as e:
                logger.error(f"Error testing strategy {strategy_type}: {str(e)}")
                success = False
                
        return success
        
    def test_near_miss_identification(self):
        """Test near-miss candidate identification logic"""
        logger.info("Testing near-miss candidate identification")
        
        # Set thresholds for testing
        thresholds = {
            "min_sharpe_ratio": 1.5,
            "min_profit_factor": 1.8,
            "max_drawdown": 15.0,
            "min_win_rate": 55.0
        }
        
        # Create candidates with different performance profiles
        candidates = []
        
        # Top performer (above all thresholds)
        candidates.append({
            "performance_metrics": {
                "sharpe_ratio": 1.8,
                "profit_factor": 2.0,
                "max_drawdown": 10.0,
                "win_rate": 60.0
            },
            "expected_result": "top"  # Should be classified as top performer
        })
        
        # Near-miss (within 85% of thresholds)
        candidates.append({
            "performance_metrics": {
                "sharpe_ratio": 1.3,  # Below by ~13%
                "profit_factor": 1.6,  # Below by ~11% 
                "max_drawdown": 17.0,  # Worse by ~13%
                "win_rate": 50.0       # Below by ~9%
            },
            "expected_result": "near-miss"  # Should be classified as near-miss
        })
        
        # Poor performer (below 85% of thresholds)
        candidates.append({
            "performance_metrics": {
                "sharpe_ratio": 0.9,
                "profit_factor": 1.2,
                "max_drawdown": 25.0, 
                "win_rate": 40.0
            },
            "expected_result": "poor"  # Should be classified as poor performer
        })
        
        # Edge case - exactly at 85% threshold
        candidates.append({
            "performance_metrics": {
                "sharpe_ratio": 1.275,  # Exactly 85% of 1.5
                "profit_factor": 1.53,   # Exactly 85% of 1.8
                "max_drawdown": 17.25,   # 15% worse than 15.0
                "win_rate": 46.75        # Exactly 85% of 55.0
            },
            "expected_result": "near-miss"  # Should be classified as near-miss
        })
        
        # Test the _is_near_miss_candidate method
        success = True
        for i, candidate_data in enumerate(candidates):
            metrics = candidate_data["performance_metrics"]
            expected = candidate_data["expected_result"]
            
            # Call the private method directly for testing
            is_top = self.engine._meets_performance_thresholds(metrics, thresholds)
            is_near_miss = self.engine._is_near_miss_candidate(metrics, thresholds)
            
            # Determine actual result
            if is_top:
                actual = "top"
            elif is_near_miss:
                actual = "near-miss"
            else:
                actual = "poor"
                
            logger.info(f"Candidate {i+1}: expected={expected}, actual={actual}")
            
            if expected != actual:
                logger.error(f"Candidate {i+1} misclassified: expected {expected}, got {actual}")
                success = False
                
        return success
        
    def test_event_emission(self):
        """Test that events are properly emitted during processing"""
        logger.info("Testing event emission")
        
        # Clear previous events
        self.event_listener.event_log = []
        
        # Run a short test process
        self.engine.start_process(
            universe="TEST",
            strategy_types=["Iron Condor", "Strangle", "Butterfly Spread"],
            thresholds={
                "min_sharpe_ratio": 1.5,
                "min_profit_factor": 1.8,
                "max_drawdown": 15.0,
                "min_win_rate": 55.0
            }
        )
        
        # Wait for process to complete (up to 30 seconds)
        import time
        max_wait = 30
        start_time = time.time()
        
        while self.engine.is_running and time.time() - start_time < max_wait:
            time.sleep(1)
            
        # If still running, stop it
        if self.engine.is_running:
            self.engine.stop_process()
            
        # Get event summary
        event_summary = self.event_listener.get_event_summary()
        logger.info(f"Event summary: {event_summary}")
        
        # Verify we received the expected event types
        expected_events = [
            EventType.STRATEGY_GENERATED,
            EventType.STRATEGY_EVALUATED,
            EventType.STRATEGY_OPTIMISED
        ]
        
        success = True
        for event_type in expected_events:
            if event_type not in event_summary or event_summary[event_type] == 0:
                logger.error(f"Missing expected events of type: {event_type}")
                success = False
        
        # Check content of optimization events specifically
        optimization_events = self.event_listener.get_events_by_type(EventType.STRATEGY_OPTIMISED)
        
        if not optimization_events:
            logger.error("No optimization events received")
            success = False
        else:
            for event in optimization_events:
                data = event["data"]
                required_fields = ["strategy_id", "strategy_type", "before_metrics", "after_metrics"]
                
                for field in required_fields:
                    if field not in data:
                        logger.error(f"Optimization event missing required field: {field}")
                        success = False
            
        return success
        
    def run_all_tests(self):
        """Run all tests and return overall status"""
        logger.info("Starting core autonomous engine tests")
        
        tests = [
            ("Strategy Adapter Integration", self.test_strategy_adapters),
            ("Near-Miss Identification", self.test_near_miss_identification),
            ("Event Emission", self.test_event_emission)
        ]
        
        # Run each test and collect results
        results = []
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                success = test_func()
                results.append((test_name, success))
                logger.info(f"Test {test_name}: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                logger.error(f"Error in test {test_name}: {str(e)}")
                results.append((test_name, False))
        
        # Restore original engine methods
        self.restore_engine_methods()
        
        # Generate summary
        logger.info("Test results summary:")
        all_passed = True
        for test_name, success in results:
            logger.info(f"- {test_name}: {'PASS' if success else 'FAIL'}")
            if not success:
                all_passed = False
                
        return all_passed
            

def main():
    print("\n" + "="*60)
    print("AUTONOMOUS ENGINE CORE FUNCTIONALITY TEST")
    print("="*60)
    
    test = MockedStrategyTest()
    success = test.run_all_tests()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Overall test result: {'PASS' if success else 'FAIL'}")
    print(f"See autonomous_core_test.log for detailed results")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
