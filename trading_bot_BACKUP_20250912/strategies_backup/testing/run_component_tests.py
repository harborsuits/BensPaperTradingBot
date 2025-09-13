#!/usr/bin/env python
"""
Component Test Runner

This script runs the component tests independently from the Streamlit UI.
It can be executed directly from the command line to test all components
or specific component types.
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from trading_bot.strategies.testing.component_tests import run_all_tests
    from trading_bot.strategies.testing.component_tester import ComponentTestRunner
    from trading_bot.strategies.components.signal_generators import (
        MovingAverageSignalGenerator, RSISignalGenerator, BollingerBandSignalGenerator
    )
    from trading_bot.strategies.components.filters import (
        VolumeFilter, VolatilityFilter, TimeOfDayFilter
    )
    from trading_bot.strategies.components.position_sizers import (
        FixedRiskPositionSizer, VolatilityAdjustedPositionSizer
    )
    from trading_bot.strategies.components.exit_managers import (
        TrailingStopExitManager, TakeProfitExitManager
    )
    logger.info("Successfully imported component test modules")
except ImportError as e:
    logger.error(f"Error importing component test modules: {e}")
    raise

def parse_arguments():
    """Parse command line arguments for the test runner."""
    parser = argparse.ArgumentParser(description="Run component tests for the modular strategy system")
    
    parser.add_argument("--component-type", type=str, choices=["signal", "filter", "position", "exit", "all"],
                      default="all", help="Type of component to test")
    
    parser.add_argument("--component-name", type=str, 
                      help="Specific component name to test (e.g., MovingAverageSignalGenerator)")
    
    parser.add_argument("--save-results", action="store_true", 
                      help="Save test results to file")
    
    parser.add_argument("--results-dir", type=str, default="test_results",
                      help="Directory to save test results")
    
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()

def main():
    """Main function to run component tests."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting component tests for {args.component_type}")
    
    # Create test runner
    test_runner = ComponentTestRunner()
    
    # Run tests based on component type
    if args.component_type == "all" and not args.component_name:
        logger.info("Running all component tests")
        results = run_all_tests()
    else:
        # Run specific component tests
        tests_to_run = []
        
        if args.component_name:
            logger.info(f"Running tests for specific component: {args.component_name}")
            # Import the specific component test
            try:
                from trading_bot.strategies.testing.component_tests import (
                    SignalGeneratorTests,
                    FilterTests,
                    PositionSizerTests,
                    ExitManagerTests
                )
                
                if "SignalGenerator" in args.component_name:
                    test_class = SignalGeneratorTests
                elif "Filter" in args.component_name:
                    test_class = FilterTests
                elif "PositionSizer" in args.component_name:
                    test_class = PositionSizerTests
                elif "ExitManager" in args.component_name:
                    test_class = ExitManagerTests
                else:
                    logger.error(f"Unknown component name: {args.component_name}")
                    return
                
                # Create test instance
                test_instance = test_class()
                
                # Get test method
                test_method_name = f"test_{args.component_name.replace('SignalGenerator', '').replace('Filter', '').replace('PositionSizer', '').replace('ExitManager', '').lower()}"
                test_method = getattr(test_instance, test_method_name, None)
                
                if test_method:
                    test_method()
                    logger.info(f"Completed test for {args.component_name}")
                else:
                    logger.error(f"No test method found for {args.component_name}")
                    
            except (ImportError, AttributeError) as e:
                logger.error(f"Error running specific component test: {e}")
                return
        else:
            # Run tests for a component type
            logger.info(f"Running tests for component type: {args.component_type}")
            component_type_map = {
                "signal": "SignalGeneratorTests",
                "filter": "FilterTests",
                "position": "PositionSizerTests",
                "exit": "ExitManagerTests"
            }
            
            try:
                from trading_bot.strategies.testing.component_tests import (
                    SignalGeneratorTests,
                    FilterTests,
                    PositionSizerTests,
                    ExitManagerTests
                )
                
                if args.component_type == "signal":
                    test_class = SignalGeneratorTests
                elif args.component_type == "filter":
                    test_class = FilterTests
                elif args.component_type == "position":
                    test_class = PositionSizerTests
                elif args.component_type == "exit":
                    test_class = ExitManagerTests
                
                # Create test instance
                test_instance = test_class()
                
                # Get all test methods
                test_methods = [method for method in dir(test_instance) if method.startswith("test_")]
                
                for method_name in test_methods:
                    test_method = getattr(test_instance, method_name)
                    test_method()
                    logger.info(f"Completed test: {method_name}")
                
            except (ImportError, AttributeError) as e:
                logger.error(f"Error running component type tests: {e}")
                return
    
    # Save results if requested
    if args.save_results:
        results_dir = os.path.join(os.path.dirname(__file__), args.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        component_type = args.component_type
        component_name = args.component_name or "all"
        
        results_file = os.path.join(results_dir, f"test_results_{component_type}_{component_name}_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Test results saved to {results_file}")
    
    logger.info("Component tests completed")

if __name__ == "__main__":
    main()
