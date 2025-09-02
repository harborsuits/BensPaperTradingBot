#!/usr/bin/env python3
"""
End-to-End Tests for Real Strategy Integration

This script tests the integration of real strategy implementations with the
strategy adapter and autonomous engine, using the market data generator
for simulated test data.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Type
import pandas as pd
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("strategy_integration")

# Add project root to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components for testing
from trading_bot.strategies.strategy_adapter import StrategyAdapter, create_strategy_adapter
from trading_bot.strategies.components.component_registry import ComponentRegistry
from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.event_system import EventBus, EventType
from trading_bot.testing.market_data_generator import MarketDataGenerator

class EventCollector:
    """Collects and tracks events for testing."""
    
    def __init__(self):
        self.events = []
        self.event_counts = {}
        self.register_handlers()
        
    def register_handlers(self):
        """Register for all event types of interest."""
        event_bus = EventBus()
        
        # Event types to track
        event_types = [
            EventType.STRATEGY_GENERATED,
            EventType.STRATEGY_EVALUATED,
            EventType.STRATEGY_OPTIMISED,
            EventType.STRATEGY_EXHAUSTED
        ]
        
        for event_type in event_types:
            event_bus.register(event_type, self.handle_event)
            
    def handle_event(self, event_type, event_data):
        """Handle an event by recording it."""
        logger.info(f"Event received: {event_type}")
        
        # Record the event
        self.events.append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.datetime.now()
        })
        
        # Update counts
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
        else:
            self.event_counts[event_type] = 1
    
    def get_events_of_type(self, event_type):
        """Get all events of a specific type."""
        return [event for event in self.events if event["type"] == event_type]
    
    def get_event_summary(self):
        """Get a summary of events received."""
        return self.event_counts
    
    def clear(self):
        """Clear all tracked events."""
        self.events = []
        self.event_counts = {}


class StrategyIntegrationTest:
    """Test the integration of real strategies with the adapter and engine."""
    
    def __init__(self):
        """Initialize the integration test."""
        # Components
        self.registry = ComponentRegistry()
        self.engine = AutonomousEngine()
        self.event_collector = EventCollector()
        self.data_generator = MarketDataGenerator(seed=42)
        
        # Test data
        self.test_data = None
        self.strategy_types = []
        
        # Test parameters
        self.thresholds = {
            "min_sharpe_ratio": 1.5,
            "min_profit_factor": 1.8,
            "max_drawdown": 15.0,
            "min_win_rate": 55.0
        }
    
    def setup(self):
        """Set up the test environment."""
        logger.info("Setting up strategy integration test")
        
        # Generate test market data
        logger.info("Generating test market data")
        self.test_data = self.data_generator.generate_test_dataset(
            num_stocks=3,
            days=120,
            include_options=True
        )
        
        # Get all registered strategy types
        self.strategy_types = self.registry.get_registered_strategy_types()
        logger.info(f"Found {len(self.strategy_types)} registered strategy types")
        
        if not self.strategy_types:
            logger.error("No strategy types registered")
            return False
            
        # Log found strategy types
        for strategy_type in self.strategy_types:
            logger.info(f"Found strategy type: {strategy_type}")
            
        return True
    
    def test_strategy_adapters(self):
        """Test that all registered strategies can be wrapped with adapters."""
        logger.info("Testing strategy adapters")
        
        failed_strategies = []
        adapted_strategies = []
        
        for strategy_type in self.strategy_types:
            try:
                # Get strategy instance through the registry
                strategy = self.registry.get_strategy_instance(strategy_type)
                
                if not strategy:
                    logger.error(f"Failed to instantiate strategy: {strategy_type}")
                    failed_strategies.append({
                        "type": strategy_type,
                        "error": "Failed to instantiate"
                    })
                    continue
                
                # Check if it has the required interface methods
                required_methods = ['generate_signals', 'size_position', 'manage_open_trades']
                missing_methods = []
                
                for method in required_methods:
                    if not hasattr(strategy, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    logger.error(f"Strategy {strategy_type} missing required methods: {missing_methods}")
                    failed_strategies.append({
                        "type": strategy_type,
                        "error": f"Missing methods: {missing_methods}"
                    })
                    continue
                
                # Strategy has all required methods
                logger.info(f"Strategy {strategy_type} successfully adapted")
                adapted_strategies.append(strategy_type)
                
            except Exception as e:
                logger.error(f"Error testing strategy {strategy_type}: {str(e)}")
                failed_strategies.append({
                    "type": strategy_type,
                    "error": str(e)
                })
        
        # Log results
        logger.info(f"Successfully adapted strategies: {len(adapted_strategies)} / {len(self.strategy_types)}")
        if failed_strategies:
            logger.error(f"Failed strategies: {len(failed_strategies)}")
            for failed in failed_strategies:
                logger.error(f"  {failed['type']}: {failed['error']}")
        
        return len(failed_strategies) == 0
    
    def test_signal_generation(self):
        """Test signal generation with adapted strategies using test data."""
        logger.info("Testing signal generation with adapted strategies")
        
        success_count = 0
        failure_count = 0
        results = []
        
        for strategy_type in self.strategy_types:
            try:
                # Get strategy instance
                strategy = self.registry.get_strategy_instance(strategy_type)
                
                if not strategy:
                    logger.error(f"Failed to instantiate strategy: {strategy_type}")
                    failure_count += 1
                    continue
                
                # Prepare test data for this strategy
                # Simplify by using the first stock's data
                if 'stocks' in self.test_data:
                    test_symbol = list(self.test_data['stocks'].keys())[0]
                    market_data = {
                        test_symbol: self.test_data['stocks'][test_symbol]
                    }
                    
                    # Add options data if available and strategy might use it
                    if 'options' in self.test_data and test_symbol in self.test_data['options']:
                        market_data['options'] = {
                            test_symbol: self.test_data['options'][test_symbol]
                        }
                else:
                    logger.error("Test data does not contain stock data")
                    failure_count += 1
                    continue
                
                # Generate signals
                logger.info(f"Generating signals for {strategy_type}")
                signals = strategy.generate_signals(market_data)
                
                # Check if signals were generated
                if signals and isinstance(signals, list):
                    logger.info(f"Strategy {strategy_type} generated {len(signals)} signals")
                    success_count += 1
                    results.append({
                        "type": strategy_type,
                        "signals_count": len(signals),
                        "first_signal": signals[0] if signals else None
                    })
                else:
                    logger.warning(f"Strategy {strategy_type} did not generate any signals")
                    failure_count += 1
                
            except Exception as e:
                logger.error(f"Error testing signal generation for {strategy_type}: {str(e)}")
                failure_count += 1
        
        # Log results
        logger.info(f"Signal generation test results: {success_count} succeeded, {failure_count} failed")
        for result in results:
            logger.info(f"  {result['type']}: {result['signals_count']} signals")
            if result['first_signal']:
                logger.info(f"    First signal: {result['first_signal']}")
        
        return failure_count == 0
    
    def test_engine_integration(self):
        """Test integration with the autonomous engine using one strategy type."""
        if not self.strategy_types:
            logger.error("No strategy types available for testing")
            return False
        
        # Select the first strategy type for testing
        test_strategy_type = self.strategy_types[0]
        logger.info(f"Testing engine integration with strategy type: {test_strategy_type}")
        
        # Configure the test to use just this strategy
        try:
            # Patch the engine's _scan_market method to use our test data
            original_scan_market = self.engine._scan_market
            
            def mock_scan_market(universe, strategy_type):
                """Return test symbols for our test strategy."""
                return list(self.test_data['stocks'].keys())
                
            self.engine._scan_market = mock_scan_market
            
            # Patch the engine's _backtest_candidate method
            original_backtest = self.engine._backtest_candidate
            
            def mock_backtest(candidate):
                """Mock backtesting to return predefined metrics."""
                logger.info(f"Mock backtesting candidate: {candidate.strategy_id}")
                
                # Set metrics that will trigger optimization
                candidate.performance_metrics = {
                    "sharpe_ratio": 1.3,  # Below threshold but within 15%
                    "profit_factor": 1.6,  # Below threshold but within 15%
                    "max_drawdown": 17.0,  # Worse than threshold but within 15%
                    "win_rate": 50.0       # Below threshold but within 15%
                }
                
                candidate.status = "backtested"
                return True
                
            self.engine._backtest_candidate = mock_backtest
            
            # Clear previous events
            self.event_collector.clear()
            
            # Start the process
            logger.info("Starting autonomous process with test strategy")
            self.engine.start_process(
                universe="TEST",
                strategy_types=[test_strategy_type],
                thresholds=self.thresholds
            )
            
            # Wait for process to complete
            import time
            max_wait = 30  # seconds
            start_time = time.time()
            
            logger.info(f"Waiting for process to complete (timeout: {max_wait}s)")
            while self.engine.is_running and time.time() - start_time < max_wait:
                # Check progress periodically
                status = self.engine.get_status()
                logger.info(f"Progress: {status.get('progress', 0)}% - {status.get('status_message', '')}")
                time.sleep(1)
                
            # Stop the process if still running
            if self.engine.is_running:
                logger.warning("Process did not complete within timeout, stopping")
                self.engine.stop_process()
            
            # Analyze events
            event_summary = self.event_collector.get_event_summary()
            logger.info(f"Event summary: {event_summary}")
            
            # Check for specific events
            generated_events = self.event_collector.get_events_of_type(EventType.STRATEGY_GENERATED)
            evaluated_events = self.event_collector.get_events_of_type(EventType.STRATEGY_EVALUATED)
            optimized_events = self.event_collector.get_events_of_type(EventType.STRATEGY_OPTIMISED)
            
            logger.info(f"Generated events: {len(generated_events)}")
            logger.info(f"Evaluated events: {len(evaluated_events)}")
            logger.info(f"Optimized events: {len(optimized_events)}")
            
            # Check if optimization occurred
            if optimized_events:
                logger.info("Strategy optimization occurred as expected")
                for event in optimized_events:
                    before = event["data"].get("before_metrics", {})
                    after = event["data"].get("after_metrics", {})
                    logger.info(f"Optimization improved sharpe from {before.get('sharpe_ratio')} to {after.get('sharpe_ratio')}")
                
                return True
            else:
                logger.error("No optimization events detected")
                return False
                
        except Exception as e:
            logger.error(f"Error in engine integration test: {str(e)}")
            return False
        finally:
            # Restore original methods
            if hasattr(self.engine, '_scan_market') and original_scan_market:
                self.engine._scan_market = original_scan_market
            if hasattr(self.engine, '_backtest_candidate') and original_backtest:
                self.engine._backtest_candidate = original_backtest
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting strategy integration tests")
        
        # Set up
        if not self.setup():
            logger.error("Setup failed")
            return False
        
        # Run tests
        tests = [
            ("Strategy Adapters", self.test_strategy_adapters),
            ("Signal Generation", self.test_signal_generation),
            ("Engine Integration", self.test_engine_integration)
        ]
        
        # Track results
        results = []
        all_passed = True
        
        for name, test_func in tests:
            logger.info(f"Running test: {name}")
            try:
                success = test_func()
                results.append({"name": name, "success": success})
                if not success:
                    all_passed = False
                logger.info(f"Test {name}: {'PASSED' if success else 'FAILED'}")
            except Exception as e:
                logger.error(f"Error in test {name}: {str(e)}")
                results.append({"name": name, "success": False, "error": str(e)})
                all_passed = False
        
        # Print summary
        logger.info("Test results summary:")
        for result in results:
            status = "PASSED" if result["success"] else "FAILED"
            error = f" - Error: {result.get('error')}" if not result["success"] and "error" in result else ""
            logger.info(f"  {result['name']}: {status}{error}")
        
        logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed


def main():
    """Main entry point for strategy integration tests."""
    print("\n" + "="*70)
    print("STRATEGY INTEGRATION TESTS")
    print("="*70)
    
    # Run tests
    test = StrategyIntegrationTest()
    success = test.run_all_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Integration tests {'PASSED' if success else 'FAILED'}")
    print("See logs for detailed results.")
    print("="*70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
