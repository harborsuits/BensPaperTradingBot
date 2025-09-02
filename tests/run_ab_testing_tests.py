#!/usr/bin/env python3
"""
A/B Testing Framework Test Runner

This script provides a direct way to test the A/B Testing Framework components
without relying on the unittest discovery mechanism, which can sometimes
have issues with project-specific import structures.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
import numpy as np

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the test class
from trading_bot.testing.test_ab_testing_framework import (
    TestABTestCore,
    TestABTestManager,
    TestABTestAnalysis,
    TestABTestingIntegration
)

if __name__ == "__main__":
    # Create a test suite with our test classes
    test_suite = unittest.TestSuite()
    
    # Add tests from each test class
    test_suite.addTest(unittest.makeSuite(TestABTestCore))
    test_suite.addTest(unittest.makeSuite(TestABTestManager))
    test_suite.addTest(unittest.makeSuite(TestABTestAnalysis))
    test_suite.addTest(unittest.makeSuite(TestABTestingIntegration))
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(test_suite)
    
    # Report results
    print(f"\nTest Results:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Exit with appropriate code
    sys.exit(len(result.failures) + len(result.errors))
