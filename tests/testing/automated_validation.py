#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Validation Suite for Pattern-Signal Integration

This module provides automated test suites and validation workflows to ensure
the trading system correctly processes signals, confirms patterns, and generates
appropriate trades.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import concurrent.futures

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import testing components
from testing.end_to_end_tester import SignalTester, TestResult
from testing.test_signal_sets import (
    TestSignalFactory, MarketConditionSignals, 
    AssetClassSignals, generate_multi_asset_test_set
)

# Import trading bot modules
from examples.pattern_signal_integration import PatternSignalIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutomatedValidation")


class TestSuite:
    """Base class for test suites."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the test suite.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.tester = SignalTester(config)
        self.results = []
    
    def run(self) -> List[TestResult]:
        """
        Run the test suite.
        
        Returns:
            List of test results
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def generate_report(self) -> str:
        """Generate a report of the test results."""
        return self.tester.generate_report()
    
    def save_results(self, filename: str):
        """Save results to a file."""
        self.tester.save_results(filename)


class BasicSignalSuite(TestSuite):
    """Basic signal processing test suite."""
    
    def run(self) -> List[TestResult]:
        """
        Run basic signal tests for all sources.
        
        Returns:
            List of test results
        """
        logger.info("Running Basic Signal Suite")
        
        # Test TradingView signals
        tradingview_signal = TestSignalFactory.tradingview_entry_signal(
            "EURUSD", "buy", 1.05, timeframe="1h"
        )
        result1 = self.tester.run_basic_test(tradingview_signal, "tradingview")
        self.results.append(result1)
        
        # Test Alpaca signals
        alpaca_signal = TestSignalFactory.alpaca_order_filled(
            "AAPL", "buy", 175.50, quantity=10
        )
        result2 = self.tester.run_basic_test(alpaca_signal, "alpaca")
        self.results.append(result2)
        
        # Test Finnhub signals
        finnhub_signal = TestSignalFactory.finnhub_trade(
            "MSFT", 280.75, volume=100
        )
        result3 = self.tester.run_basic_test(finnhub_signal, "finnhub")
        self.results.append(result3)
        
        logger.info(f"Basic Signal Suite completed: {len(self.results)} tests run")
        return self.results


class PatternConfirmationSuite(TestSuite):
    """Test suite for pattern confirmation scenarios."""
    
    def run(self) -> List[TestResult]:
        """
        Run pattern confirmation tests.
        
        Returns:
            List of test results
        """
        logger.info("Running Pattern Confirmation Suite")
        
        # Test pattern confirmation with TradingView signal
        tv_signal = TestSignalFactory.tradingview_entry_signal(
            "EURUSD", "buy", 1.05, timeframe="1h"
        )
        pattern = TestSignalFactory.pattern_detection(
            "EURUSD", "pin_bar", "long", 1.05, confidence=0.85
        )
        result1 = self.tester.run_pattern_confirmation_test(tv_signal, pattern)
        self.results.append(result1)
        
        # Test pattern confirmation with Alpaca signal
        alpaca_signal = TestSignalFactory.alpaca_order_filled(
            "AAPL", "buy", 175.50, quantity=10
        )
        pattern2 = TestSignalFactory.pattern_detection(
            "AAPL", "engulfing", "long", 175.50, confidence=0.85
        )
        result2 = self.tester.run_pattern_confirmation_test(alpaca_signal, pattern2, source="alpaca")
        self.results.append(result2)
        
        # Test pattern confirmation with opposite direction (should fail)
        tv_signal3 = TestSignalFactory.tradingview_entry_signal(
            "GBPUSD", "buy", 1.25, timeframe="1h"
        )
        pattern3 = TestSignalFactory.pattern_detection(
            "GBPUSD", "pin_bar", "short", 1.25, confidence=0.85
        )
        result3 = self.tester.run_pattern_confirmation_test(tv_signal3, pattern3)
        self.results.append(result3)
        
        logger.info(f"Pattern Confirmation Suite completed: {len(self.results)} tests run")
        return self.results


class MarketConditionSuite(TestSuite):
    """Test suite for different market conditions."""
    
    def run(self) -> List[TestResult]:
        """
        Run tests for different market conditions.
        
        Returns:
            List of test results
        """
        logger.info("Running Market Condition Suite")
        
        # Test trending market
        trending_signals = MarketConditionSignals.trending_market("EURUSD", "up", 1.05)
        result1 = self.tester.run_concurrent_signals_test(trending_signals)
        self.results.append(result1)
        
        # Test ranging market
        ranging_signals = MarketConditionSignals.ranging_market("GBPUSD", 1.25)
        result2 = self.tester.run_concurrent_signals_test(ranging_signals)
        self.results.append(result2)
        
        # Test breakout scenario
        breakout_signals = MarketConditionSignals.breakout_scenario("USDJPY", 150.0, direction="up")
        result3 = self.tester.run_concurrent_signals_test(breakout_signals)
        self.results.append(result3)
        
        # Test reversal scenario
        reversal_signals = MarketConditionSignals.reversal_scenario("EURUSD", 1.05, direction="bullish")
        result4 = self.tester.run_concurrent_signals_test(reversal_signals)
        self.results.append(result4)
        
        logger.info(f"Market Condition Suite completed: {len(self.results)} tests run")
        return self.results


class AssetClassSuite(TestSuite):
    """Test suite for different asset classes."""
    
    def run(self) -> List[TestResult]:
        """
        Run tests for different asset classes.
        
        Returns:
            List of test results
        """
        logger.info("Running Asset Class Suite")
        
        # Test forex signals
        forex_signals = AssetClassSignals.forex_signals("EURUSD")
        result1 = self.tester.run_concurrent_signals_test(forex_signals)
        self.results.append(result1)
        
        # Test crypto signals
        crypto_signals = AssetClassSignals.crypto_signals("BTCUSD")
        result2 = self.tester.run_concurrent_signals_test(crypto_signals)
        self.results.append(result2)
        
        # Test stock signals
        stock_signals = AssetClassSignals.stock_signals("AAPL")
        result3 = self.tester.run_concurrent_signals_test(stock_signals)
        self.results.append(result3)
        
        logger.info(f"Asset Class Suite completed: {len(self.results)} tests run")
        return self.results


class FullSystemTest(TestSuite):
    """Comprehensive test of the entire system with many signals."""
    
    def run(self) -> List[TestResult]:
        """
        Run a full system test with many signals across different assets.
        
        Returns:
            List of test results
        """
        logger.info("Running Full System Test")
        
        # Generate a comprehensive test set
        test_signals = generate_multi_asset_test_set()
        
        # Run the test with all signals
        result = self.tester.run_concurrent_signals_test(test_signals, duration_seconds=30)
        self.results.append(result)
        
        logger.info("Full System Test completed")
        return self.results


class BoundaryConditionSuite(TestSuite):
    """Test suite for boundary conditions and error handling."""
    
    def run(self) -> List[TestResult]:
        """
        Run tests for boundary conditions and error cases.
        
        Returns:
            List of test results
        """
        logger.info("Running Boundary Condition Suite")
        
        # Test with invalid signal (missing required fields)
        invalid_signal = {"action": "buy"}  # Missing symbol
        result1 = self.tester.run_basic_test(invalid_signal, "tradingview")
        self.results.append(result1)
        
        # Test with invalid JSON format
        invalid_json = "{invalid_json: test}"
        try:
            # We need to manually create this test because run_basic_test expects valid JSON
            test_name = "Invalid JSON Test"
            self.tester.start_test(test_name)
            
            # This should fail gracefully
            try:
                webhook_url = f"http://localhost:{self.tester.integration.webhook_handler.port}/{self.tester.integration.webhook_handler.path}"
                response = requests.post(webhook_url, data=invalid_json, headers={"Content-Type": "application/json"})
                self.tester.current_test.add_event(TestEvent.SIGNAL_SENT, {"invalid_json": True})
            except Exception as e:
                self.tester.current_test.add_error(f"Error with invalid JSON: {str(e)}", e)
            
            # Monitor for errors
            self.tester.monitor_signal_flow(5)
            result2 = self.tester.end_test(False)  # Expected to fail
        except Exception as e:
            logger.error(f"Error in invalid JSON test: {str(e)}")
            result2 = None
        
        if result2:
            self.results.append(result2)
        
        # Test duplicate signals
        duplicate_signal = TestSignalFactory.tradingview_entry_signal("EURUSD", "buy", 1.05)
        
        # Send the same signal twice
        test_name = "Duplicate Signal Test"
        self.tester.start_test(test_name)
        
        try:
            # Send first signal
            response1 = self.tester.send_test_tradingview_signal(duplicate_signal)
            time.sleep(1)  # Wait a moment
            
            # Send duplicate signal
            response2 = self.tester.send_test_tradingview_signal(duplicate_signal)
            
            # Monitor signal flow
            self.tester.monitor_signal_flow(10)
            result3 = self.tester.end_test(True)
            self.results.append(result3)
        except Exception as e:
            logger.error(f"Error in duplicate signal test: {str(e)}")
        
        logger.info(f"Boundary Condition Suite completed: {len(self.results)} tests run")
        return self.results


class ValidationRunner:
    """Runs validation test suites."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the validation runner.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.suites = {}
        self.results = {}
        
        # Initialize test suites
        self._init_suites()
    
    def _init_suites(self):
        """Initialize all test suites."""
        self.suites = {
            "basic": BasicSignalSuite(self.config),
            "pattern": PatternConfirmationSuite(self.config),
            "market": MarketConditionSuite(self.config),
            "asset": AssetClassSuite(self.config),
            "full": FullSystemTest(self.config),
            "boundary": BoundaryConditionSuite(self.config)
        }
    
    def run_suite(self, suite_name: str) -> List[TestResult]:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the suite to run
            
        Returns:
            List of test results
        """
        if suite_name not in self.suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        logger.info(f"Running test suite: {suite_name}")
        results = self.suites[suite_name].run()
        self.results[suite_name] = results
        
        return results
    
    def run_all(self) -> Dict[str, List[TestResult]]:
        """
        Run all test suites.
        
        Returns:
            Dictionary of suite results
        """
        logger.info("Running all test suites")
        
        for suite_name in self.suites:
            self.run_suite(suite_name)
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of all test results.
        
        Returns:
            Text report
        """
        if not self.results:
            return "No tests have been run"
        
        lines = [
            "=" * 80,
            "                   AUTOMATED VALIDATION REPORT                   ",
            "=" * 80,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Suites Run: {len(self.results)}",
            "=" * 80,
            ""
        ]
        
        # Summary of all suites
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for suite_name, results in self.results.items():
            suite_tests = len(results)
            suite_passed = sum(1 for r in results if r.success)
            suite_failed = sum(1 for r in results if not r.success)
            
            total_tests += suite_tests
            total_passed += suite_passed
            total_failed += suite_failed
            
            lines.append(f"Suite: {suite_name}")
            lines.append(f"  Tests: {suite_tests}, Passed: {suite_passed}, Failed: {suite_failed}")
        
        lines.append("")
        lines.append("Summary:")
        lines.append(f"  Total Tests: {total_tests}")
        lines.append(f"  Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        lines.append(f"  Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
        lines.append("")
        
        # Detailed report for each suite
        for suite_name, results in self.results.items():
            lines.append("=" * 60)
            lines.append(f"Suite: {suite_name}")
            lines.append("=" * 60)
            
            for i, result in enumerate(results, 1):
                lines.append(f"Test #{i}: {result.test_name}")
                lines.append(f"Status: {'SUCCESS' if result.success else 'FAILURE'}")
                
                # Key metrics
                for key, value in result.metrics.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.2f}")
                    else:
                        lines.append(f"  {key}: {value}")
                
                # Errors
                if result.errors:
                    lines.append("  Errors:")
                    for error in result.errors[:3]:  # Show first 3 errors
                        lines.append(f"    - {error['message']}")
                    
                    if len(result.errors) > 3:
                        lines.append(f"    ... {len(result.errors) - 3} more errors")
                
                lines.append("-" * 40)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_report(self, filename: str):
        """
        Save the report to a file.
        
        Args:
            filename: Output filename
        """
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
            
        logger.info(f"Validation report saved to {filename}")
    
    def save_all_results(self, directory: str):
        """
        Save all test results to JSON files.
        
        Args:
            directory: Output directory
        """
        os.makedirs(directory, exist_ok=True)
        
        for suite_name, suite in self.suites.items():
            if suite_name in self.results:
                filename = os.path.join(directory, f"{suite_name}_results.json")
                suite.save_results(filename)
        
        # Save combined report
        report_file = os.path.join(directory, "validation_report.txt")
        self.save_report(report_file)
        
        logger.info(f"All test results saved to {directory}")


def main():
    """Run the automated validation."""
    parser = argparse.ArgumentParser(description="Run automated validation tests")
    parser.add_argument("--suite", type=str, choices=["basic", "pattern", "market", "asset", "full", "boundary", "all"],
                      default="all", help="Test suite to run (default: all)")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="validation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Default config if not specified
    if not config:
        config = {
            "webhook_port": 5000,
            "webhook_path": "webhook",
            "keep_running": False,
            "pattern_strategy_config": {
                "confidence_threshold": 0.7,
                "lookback_periods": 20,
                "confirmation_required": True
            }
        }
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run validation
    runner = ValidationRunner(config)
    
    try:
        if args.suite == "all":
            runner.run_all()
        else:
            runner.run_suite(args.suite)
        
        # Generate and print report
        report = runner.generate_report()
        print(report)
        
        # Save results
        runner.save_all_results(args.output)
        
    except KeyboardInterrupt:
        print("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Error in validation: {str(e)}")
    
    print(f"Validation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
