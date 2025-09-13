#!/usr/bin/env python3
"""
Run all tests for the trading bot
"""
import os
import sys
import argparse
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_results.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TestRunner")

def run_api_tests(test_mode=False):
    """Run API endpoint tests"""
    logger.info("Running API endpoint tests...")
    
    env = os.environ.copy()
    if test_mode:
        env["TEST_MODE"] = "true"
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_api_endpoints.py", "-v"]
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        logger.info(f"API tests completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"API tests failed")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def run_backtest(symbol="AAPL"):
    """Run backtest for a symbol"""
    logger.info(f"Running backtest for {symbol}...")
    
    # Import the backtest module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from tests.test_backtest import test_backtest_functionality
        
        metrics = test_backtest_functionality()
        logger.info(f"Backtest completed with metrics: {metrics}")
        return True
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return False

def run_integration_tests(test_mode=False):
    """Run integration tests"""
    logger.info("Running integration tests...")
    
    env = os.environ.copy()
    if test_mode:
        env["TEST_MODE"] = "true"
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_integration.py", "-v"]
    
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        logger.info(f"Integration tests completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Integration tests failed")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def run_stress_test(duration=60, users=10, spawn_rate=1):
    """Run stress test using locust"""
    logger.info(f"Running stress test with {users} users for {duration} seconds...")
    
    # Check if locust is installed
    try:
        subprocess.run(["locust", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Locust not installed. Install with: pip install locust")
        return False
    
    cmd = [
        "locust", 
        "-f", "tests/locustfile.py",
        "--headless",
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--host", os.environ.get("API_BASE_URL", "http://localhost:5000")
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Stress test completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Stress test failed")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run trading bot tests")
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode for trade execution"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API endpoint tests"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests"
    )
    
    parser.add_argument(
        "--stress",
        action="store_true", 
        help="Run stress tests"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    parser.add_argument(
        "--stress-users",
        type=int,
        default=10,
        help="Number of users for stress test"
    )
    
    parser.add_argument(
        "--stress-duration",
        type=int,
        default=60,
        help="Duration in seconds for stress test"
    )
    
    args = parser.parse_args()
    
    # Run all tests if no specific test is selected
    run_all = args.all or not (args.api or args.backtest or args.integration or args.stress)
    
    if args.api or run_all:
        run_api_tests(args.test_mode)
    
    if args.backtest or run_all:
        run_backtest()
    
    if args.integration or run_all:
        run_integration_tests(args.test_mode)
    
    if args.stress or run_all:
        run_stress_test(args.stress_duration, args.stress_users)
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main() 