#!/usr/bin/env python3
"""
Test runner script for the trading_bot package.
This script discovers and runs all tests in the trading_bot/tests directory.
"""

import os
import sys
import unittest
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestRunner")

def discover_and_run_tests(pattern=None, verbosity=1, specific_test=None, include_slow=False):
    """
    Discover and run tests with specified options.
    
    Args:
        pattern: String pattern for test file names
        verbosity: Verbosity level for test output
        specific_test: Run a specific test module or path
        include_slow: Include tests marked as slow
    
    Returns:
        Test result object containing test outcomes
    """
    start_time = datetime.now()
    logger.info("Starting test suite at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Set up test pattern based on parameters
    if not pattern:
        pattern = "test_*.py"
    
    # Set up the test loader
    loader = unittest.TestLoader()
    
    # Set up the test suite
    if specific_test:
        logger.info("Running specific test: %s", specific_test)
        if os.path.isfile(specific_test):
            # Load tests from a specific file
            test_dir = os.path.dirname(specific_test)
            test_file = os.path.basename(specific_test)
            suite = loader.discover(test_dir, pattern=test_file)
        else:
            # Assume it's a module name
            try:
                suite = loader.loadTestsFromName(specific_test)
            except (ImportError, AttributeError):
                logger.error("Could not find specified test: %s", specific_test)
                return None
    else:
        # Discover all tests
        test_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(test_dir, pattern=pattern)
    
    # Filter slow tests if not included
    if not include_slow:
        filtered_suite = unittest.TestSuite()
        for test_suite in suite:
            for test_case in test_suite:
                if hasattr(test_case, '_testMethodName'):
                    test_method = getattr(test_case, test_case._testMethodName)
                    if not hasattr(test_method, 'slow') or not getattr(test_method, 'slow'):
                        filtered_suite.addTest(test_case)
        suite = filtered_suite
    
    # Count the tests
    test_count = suite.countTestCases()
    logger.info("Discovered %d test cases", test_count)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Calculate and display results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("Test suite completed in %.2f seconds", duration)
    logger.info("Ran %d tests", result.testsRun)
    logger.info("Results: %d passed, %d failed, %d errors, %d skipped", 
                result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                len(result.failures),
                len(result.errors),
                len(result.skipped))
    
    return result

def mark_test_slow(test_func):
    """
    Decorator to mark a test as slow, so it can be skipped in quick test runs.
    
    Args:
        test_func: The test function to mark
        
    Returns:
        The decorated test function
    """
    test_func.slow = True
    return test_func

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run trading_bot test suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-p", "--pattern", type=str, help="Pattern for test discovery")
    parser.add_argument("-t", "--test", type=str, help="Run a specific test module or file")
    parser.add_argument("--include-slow", action="store_true", help="Include tests marked as slow")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    # Run tests
    result = discover_and_run_tests(
        pattern=args.pattern,
        verbosity=verbosity,
        specific_test=args.test,
        include_slow=args.include_slow
    )
    
    # Set exit code based on test results
    if result and result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1) 