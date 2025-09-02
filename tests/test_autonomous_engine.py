#!/usr/bin/env python3
"""
End-to-End Test for Autonomous Trading Engine

This script performs a complete end-to-end test of the autonomous trading engine
without any UI dependencies. It validates:
1. Strategy adapter integration
2. Component registry functionality
3. Near-miss strategy identification
4. Optimization process
5. Event emission and handling

Usage:
    python test_autonomous_engine.py
"""

import os
import sys
import time
import logging
import datetime
from typing import Dict, List, Any, Set
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("autonomous_test")

# Import trading system components
from trading_bot.autonomous.autonomous_engine import AutonomousEngine
from trading_bot.event_system import EventBus, EventHandler, EventType
from trading_bot.strategies.components.component_registry import ComponentRegistry
from trading_bot.strategies.strategy_adapter import StrategyAdapter, create_strategy_adapter
from trading_bot.event_system.strategy_optimization_handlers import StrategyOptimizationTracker
from trading_bot.strategies.optimizer.enhanced_optimizer import EnhancedOptimizer

class TestEventListener:
    """Test class to listen for and log events from the autonomous engine."""
    
    def __init__(self):
        self.events_received = []
        self.optimization_events = []
        self.event_counts = {}
        self.register_handlers()
        
    def register_handlers(self):
        """Register handlers for all event types we want to track."""
        event_bus = EventBus()
        
        # Register for all event types we want to track
        for event_type in [
            EventType.MARKET_SCAN_STARTED,
            EventType.MARKET_SCAN_COMPLETED,
            EventType.STRATEGY_GENERATED,
            EventType.STRATEGY_EVALUATED,
            EventType.STRATEGY_OPTIMISED,
            EventType.STRATEGY_EXHAUSTED,
            EventType.PROCESS_COMPLETED
        ]:
            event_bus.register(event_type, self.handle_event)
            
        # Special handler for optimization events
        event_bus.register(EventType.STRATEGY_OPTIMISED, self.handle_optimization_event)
        event_bus.register(EventType.STRATEGY_EXHAUSTED, self.handle_optimization_event)
    
    def handle_event(self, event_type: EventType, event_data: Dict[str, Any]):
        """Generic event handler to log all events."""
        logger.info(f"Event received: {event_type} - {event_data}")
        self.events_received.append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.datetime.now()
        })
        
        # Track event counts
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
        else:
            self.event_counts[event_type] = 1
    
    def handle_optimization_event(self, event_type: EventType, event_data: Dict[str, Any]):
        """Special handler for optimization-related events."""
        logger.info(f"Optimization event: {event_type} - {event_data}")
        self.optimization_events.append({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.datetime.now()
        })
        
    def get_event_summary(self) -> Dict[str, int]:
        """Return a summary of events received."""
        return self.event_counts
    
    def get_optimization_results(self) -> List[Dict[str, Any]]:
        """Return all optimization events."""
        return self.optimization_events


class AutonomousEngineTest:
    """End-to-end test for the autonomous trading engine."""
    
    def __init__(self):
        """Initialize the test environment."""
        self.engine = AutonomousEngine()
        self.event_listener = TestEventListener()
        self.component_registry = ComponentRegistry()
        self.optimizer = EnhancedOptimizer()
        self.optimization_tracker = StrategyOptimizationTracker()
        
        # Set up test parameters
        self.test_universe = "TEST"
        self.strategy_types = ["Iron Condor", "Strangle", "Butterfly Spread", "Calendar Spread"]
        self.thresholds = {
            "min_sharpe_ratio": 1.5,
            "min_profit_factor": 1.8, 
            "max_drawdown": 15.0,
            "min_win_rate": 55.0
        }
        
        # Tracking test results
        self.strategies_tested = set()
        self.near_miss_candidates = []
        self.optimized_strategies = []
        self.test_start_time = None
        self.test_end_time = None
    
    def setup(self):
        """Set up the test environment."""
        logger.info("Setting up test environment")
        
        # Validate component registry has strategies
        strategies = self.component_registry.get_registered_strategy_types()
        logger.info(f"Registered strategies: {strategies}")
        
        if not strategies:
            logger.error("No strategies registered in component registry")
            return False
        
        # Verify strategy adapter functionality with a sample strategy
        sample_strategy_type = next(iter(strategies))
        strategy_instance = self.component_registry.get_strategy_instance(sample_strategy_type)
        
        if not strategy_instance:
            logger.error(f"Failed to instantiate strategy: {sample_strategy_type}")
            return False
        
        # Verify adapter wrapping
        if not isinstance(strategy_instance, StrategyAdapter) and not hasattr(strategy_instance, 'generate_signals'):
            logger.error(f"Strategy {sample_strategy_type} not properly adapted")
            return False
        
        logger.info("Test environment setup successful")
        return True
    
    def run_test(self):
        """Run the end-to-end test."""
        logger.info("Starting end-to-end test")
        self.test_start_time = datetime.datetime.now()
        
        # Step 1: Set up test environment
        if not self.setup():
            logger.error("Test setup failed")
            return False
        
        # Step 2: Start the autonomous process
        logger.info(f"Starting autonomous process with universe={self.test_universe}, "
                   f"strategy_types={self.strategy_types}, thresholds={self.thresholds}")
        
        self.engine.start_process(
            universe=self.test_universe,
            strategy_types=self.strategy_types,
            thresholds=self.thresholds
        )
        
        # Step 3: Wait for process to complete or timeout
        max_wait_time = 600  # 10 minutes
        start_time = time.time()
        
        logger.info(f"Waiting for process to complete (timeout: {max_wait_time} seconds)")
        while self.engine.is_running and time.time() - start_time < max_wait_time:
            # Get status and log progress
            status = self.engine.get_status()
            if status.get("progress", 0) % 10 == 0:  # Log every 10% progress
                logger.info(f"Progress: {status.get('progress', 0)}% - {status.get('status_message', '')}")
            time.sleep(5)
        
        # Check if process completed successfully or timed out
        if self.engine.is_running:
            logger.warning("Test timed out - force stopping")
            self.engine.stop_process()
        
        self.test_end_time = datetime.datetime.now()
        test_duration = (self.test_end_time - self.test_start_time).total_seconds()
        logger.info(f"Test completed in {test_duration:.2f} seconds")
        
        # Step 4: Analyze results
        self.analyze_results()
        
        return True
    
    def analyze_results(self):
        """Analyze the results of the test."""
        logger.info("Analyzing test results")
        
        # Get test run statistics
        candidates = self.engine.get_all_candidates()
        top_candidates = self.engine.get_top_candidates()
        
        logger.info(f"Total strategy candidates generated: {len(candidates)}")
        logger.info(f"Top candidates meeting thresholds: {len(top_candidates)}")
        
        # Analyze event data
        event_summary = self.event_listener.get_event_summary()
        logger.info(f"Event summary: {event_summary}")
        
        # Check for optimization events
        optimization_results = self.event_listener.get_optimization_results()
        logger.info(f"Total optimization events: {len(optimization_results)}")
        
        # Analyze optimization effectiveness
        optimized_ids = set()
        exhausted_ids = set()
        
        for event in optimization_results:
            event_data = event["data"]
            strategy_id = event_data.get("strategy_id")
            
            if event["type"] == EventType.STRATEGY_OPTIMISED:
                optimized_ids.add(strategy_id)
                
                # Calculate improvement percentage
                before_metrics = event_data.get("before_metrics", {})
                after_metrics = event_data.get("after_metrics", {})
                
                if before_metrics and after_metrics:
                    sharpe_before = before_metrics.get("sharpe_ratio", 0)
                    sharpe_after = after_metrics.get("sharpe_ratio", 0)
                    
                    if sharpe_before > 0:
                        improvement = ((sharpe_after - sharpe_before) / sharpe_before) * 100
                        logger.info(f"Strategy {strategy_id} improved by {improvement:.2f}%")
            
            elif event["type"] == EventType.STRATEGY_EXHAUSTED:
                exhausted_ids.add(strategy_id)
        
        logger.info(f"Successfully optimized strategies: {len(optimized_ids)}")
        logger.info(f"Strategies that could not be optimized: {len(exhausted_ids)}")
        
        # Test all strategy types
        strategy_types_seen = {c.get("strategy_type") for c in candidates if "strategy_type" in c}
        logger.info(f"Strategy types tested: {strategy_types_seen}")
        
        missing_strategy_types = set(self.strategy_types) - strategy_types_seen
        if missing_strategy_types:
            logger.warning(f"Some strategy types were not tested: {missing_strategy_types}")
        
        # Generate test report
        self.generate_test_report(
            candidates=candidates,
            top_candidates=top_candidates,
            event_summary=event_summary,
            optimization_results=optimization_results,
            optimized_ids=optimized_ids,
            exhausted_ids=exhausted_ids,
            strategy_types_seen=strategy_types_seen
        )
    
    def generate_test_report(self, **kwargs):
        """Generate a comprehensive test report."""
        logger.info("Generating test report")
        
        # Extract data from kwargs
        candidates = kwargs.get("candidates", [])
        top_candidates = kwargs.get("top_candidates", [])
        event_summary = kwargs.get("event_summary", {})
        optimization_results = kwargs.get("optimization_results", [])
        optimized_ids = kwargs.get("optimized_ids", set())
        exhausted_ids = kwargs.get("exhausted_ids", set())
        strategy_types_seen = kwargs.get("strategy_types_seen", set())
        
        # Create report sections
        report = ["# Autonomous Engine End-to-End Test Report", 
                 f"Test Run: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Duration: {(self.test_end_time - self.test_start_time).total_seconds():.2f} seconds\n"]
        
        # Summary section
        report.append("## Summary")
        report.append(f"- Total strategies generated: {len(candidates)}")
        report.append(f"- Top candidates: {len(top_candidates)}")
        report.append(f"- Strategy types tested: {', '.join(strategy_types_seen)}")
        report.append(f"- Optimization attempts: {len(optimized_ids) + len(exhausted_ids)}")
        report.append(f"- Successfully optimized: {len(optimized_ids)}")
        report.append(f"- Optimization exhausted: {len(exhausted_ids)}\n")
        
        # Event summary section
        report.append("## Event System Validation")
        for event_type, count in event_summary.items():
            report.append(f"- {event_type}: {count} events")
        report.append("")
        
        # Strategy adaptation verification
        report.append("## Strategy Adaptation Verification")
        for strategy_type in strategy_types_seen:
            count = sum(1 for c in candidates if c.get("strategy_type") == strategy_type)
            report.append(f"- {strategy_type}: {count} candidates generated")
        report.append("")
        
        # Optimization results
        if optimization_results:
            report.append("## Optimization Results")
            for event in optimization_results:
                if event["type"] == EventType.STRATEGY_OPTIMISED:
                    data = event["data"]
                    strategy_id = data.get("strategy_id", "unknown")
                    strategy_type = data.get("strategy_type", "unknown")
                    
                    before = data.get("before_metrics", {})
                    after = data.get("after_metrics", {})
                    
                    report.append(f"### Strategy {strategy_id} ({strategy_type})")
                    report.append("| Metric | Before | After | Change |")
                    report.append("|--------|--------|-------|--------|")
                    
                    for metric in ["sharpe_ratio", "profit_factor", "max_drawdown", "win_rate"]:
                        before_val = before.get(metric, 0)
                        after_val = after.get(metric, 0)
                        
                        if before_val != 0:
                            change = ((after_val - before_val) / abs(before_val)) * 100
                            change_str = f"{change:+.2f}%"
                        else:
                            change_str = "N/A"
                            
                        report.append(f"| {metric} | {before_val:.2f} | {after_val:.2f} | {change_str} |")
                    
                    report.append("")
        
        # Write report to file
        report_path = "autonomous_test_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Test report written to {report_path}")
    
    def summarize_test(self):
        """Print a brief summary of the test results."""
        event_summary = self.event_listener.get_event_summary()
        optimization_results = self.event_listener.get_optimization_results()
        
        print("\n" + "="*50)
        print("AUTONOMOUS ENGINE END-TO-END TEST SUMMARY")
        print("="*50)
        
        optimized_count = sum(1 for e in optimization_results if e["type"] == EventType.STRATEGY_OPTIMISED)
        exhausted_count = sum(1 for e in optimization_results if e["type"] == EventType.STRATEGY_EXHAUSTED)
        
        print(f"Total strategies generated: {event_summary.get(EventType.STRATEGY_GENERATED, 0)}")
        print(f"Strategies evaluated: {event_summary.get(EventType.STRATEGY_EVALUATED, 0)}")
        print(f"Successfully optimized: {optimized_count}")
        print(f"Optimization exhausted: {exhausted_count}")
        print("="*50)
        print(f"See autonomous_test.log and autonomous_test_report.md for detailed results")
        print("="*50 + "\n")


def main():
    """Main entry point for the test script."""
    logger.info("Starting autonomous engine end-to-end test")
    
    test = AutonomousEngineTest()
    test.run_test()
    test.summarize_test()
    
    logger.info("Test completed")


if __name__ == "__main__":
    main()
