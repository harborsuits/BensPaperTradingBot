#!/usr/bin/env python3
"""
Run All Tests for Autonomous Options Strategy Integration

This script runs all tests and benchmarks for the autonomous options strategy
integration, providing a comprehensive verification of functionality.
"""

import os
import sys
import logging
import time
import unittest
from datetime import datetime
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_suite.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_runner")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
import test_strategy_adapter
import test_autonomous_optimization
import test_strategy_integration
import test_risk_integration
import optimization_benchmark

def run_unit_tests():
    """Run unittest-based tests."""
    logger.info("Running unit tests")
    
    # Create test suite for unittest modules
    test_suite = unittest.TestSuite()
    
    # Add tests from test_strategy_adapter
    adapter_tests = unittest.defaultTestLoader.loadTestsFromModule(test_strategy_adapter)
    test_suite.addTest(adapter_tests)
    
    # Add tests from test_autonomous_optimization
    optimization_tests = unittest.defaultTestLoader.loadTestsFromModule(test_autonomous_optimization)
    test_suite.addTest(optimization_tests)
    
    # Add tests from test_risk_integration
    risk_tests = unittest.defaultTestLoader.loadTestsFromModule(test_risk_integration)
    test_suite.addTest(risk_tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests."""
    logger.info("Running strategy integration tests")
    
    # Create and run integration test
    integration_test = test_strategy_integration.StrategyIntegrationTest()
    success = integration_test.run_all_tests()
    
    return success

def run_performance_benchmark():
    """Run performance benchmarks."""
    logger.info("Running optimization performance benchmark")
    
    # Create and run benchmark
    benchmark = optimization_benchmark.OptimizationBenchmark()
    success = benchmark.run_benchmark()
    
    if success:
        # Generate reports
        benchmark.generate_report()
        benchmark.generate_visualizations()
    
    return success

def run_all_tests():
    """Run all tests and benchmarks."""
    logger.info("Starting complete test suite")
    start_time = time.time()
    
    # Track results
    results = {
        "unit_tests": False,
        "integration_tests": False,
        "benchmarks": False
    }
    
    # Run unit tests
    try:
        results["unit_tests"] = run_unit_tests()
    except Exception as e:
        logger.error(f"Error running unit tests: {str(e)}")
    
    # Run integration tests
    try:
        results["integration_tests"] = run_integration_tests()
    except Exception as e:
        logger.error(f"Error running integration tests: {str(e)}")
    
    # Run performance benchmark
    try:
        results["benchmarks"] = run_performance_benchmark()
    except Exception as e:
        logger.error(f"Error running performance benchmark: {str(e)}")
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    print(f"Completed in {total_time:.2f} seconds\n")
    
    print("Unit Tests:", "PASSED" if results["unit_tests"] else "FAILED")
    print("Integration Tests:", "PASSED" if results["integration_tests"] else "FAILED")
    print("Performance Benchmarks:", "PASSED" if results["benchmarks"] else "FAILED")
    
    overall = all(results.values())
    print("\nOverall Result:", "PASSED" if overall else "FAILED")
    print("="*70)
    
    return overall

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
