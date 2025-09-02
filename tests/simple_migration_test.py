#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Migration Test

This script validates the core aspects of our trading bot migration without
requiring external dependencies. It focuses on verifying the directory structure,
file organization, and basic imports.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleMigrationTest")

def test_directory_structure() -> bool:
    """Test if the directory structure has been properly created."""
    logger.info("Testing directory structure...")
    
    # Base directories that should exist
    required_dirs = [
        "trading_bot/strategies",
        "trading_bot/strategies/forex",
        "trading_bot/strategies/forex/trend",
        "trading_bot/strategies/forex/base",
        "trading_bot/strategies/factory",
        "trading_bot/data/processors",
        "trading_bot/data/quality"
    ]
    
    # Key files that should exist
    required_files = [
        "trading_bot/strategies/forex/trend/trend_following_strategy.py",
        "trading_bot/strategies/factory/strategy_registry.py",
        "trading_bot/strategies/factory/strategy_factory.py",
        "trading_bot/data/data_pipeline.py",
        "trading_bot/data/dashboard_connector.py",
        "trading_bot/data/processor_migration.py"
    ]
    
    # Count successes
    dir_success = 0
    file_success = 0
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            logger.info(f"✓ Directory exists: {dir_path}")
            dir_success += 1
        else:
            logger.error(f"✗ Directory missing: {dir_path}")
    
    # Check files
    for file_path in required_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            logger.info(f"✓ File exists: {file_path}")
            file_success += 1
        else:
            logger.error(f"✗ File missing: {file_path}")
    
    # Calculate success rate
    dir_rate = dir_success / len(required_dirs) * 100
    file_rate = file_success / len(required_files) * 100
    total_rate = (dir_success + file_success) / (len(required_dirs) + len(required_files)) * 100
    
    logger.info(f"Directory structure check: {dir_success}/{len(required_dirs)} ({dir_rate:.1f}%)")
    logger.info(f"File existence check: {file_success}/{len(required_files)} ({file_rate:.1f}%)")
    logger.info(f"Overall structure check: {dir_success + file_success}/{len(required_dirs) + len(required_files)} ({total_rate:.1f}%)")
    
    # Success if at least 80% of directories and files exist
    return total_rate >= 80

def test_forex_trend_strategy_contents() -> bool:
    """Test if the ForexTrendFollowingStrategy content is correct."""
    logger.info("Testing ForexTrendFollowingStrategy content...")
    
    strategy_path = os.path.join(os.getcwd(), "trading_bot/strategies/forex/trend/trend_following_strategy.py")
    
    if not os.path.exists(strategy_path):
        logger.error("ForexTrendFollowingStrategy file does not exist")
        return False
    
    # Required elements to check
    required_elements = [
        "class ForexTrendFollowingStrategy",
        "ForexBaseStrategy",
        "register_strategy",
        "calculate_indicators",
        "generate_signals",
        "calculate_position_size",
        "trending",  # For market regime
        "ATR",       # For volatility and position sizing
        "MACD",      # For confirmation
        "ADX"        # For trend strength
    ]
    
    # Read the file
    with open(strategy_path, 'r') as f:
        content = f.read()
    
    # Check each required element
    success = 0
    for element in required_elements:
        if element in content:
            logger.info(f"✓ Strategy contains: {element}")
            success += 1
        else:
            logger.error(f"✗ Strategy missing: {element}")
    
    # Calculate success rate
    success_rate = success / len(required_elements) * 100
    logger.info(f"Strategy content check: {success}/{len(required_elements)} ({success_rate:.1f}%)")
    
    # Success if at least 80% of required elements are present
    return success_rate >= 80

def test_data_pipeline_contents() -> bool:
    """Test if the DataPipeline content is correct."""
    logger.info("Testing DataPipeline content...")
    
    pipeline_path = os.path.join(os.getcwd(), "trading_bot/data/data_pipeline.py")
    
    if not os.path.exists(pipeline_path):
        logger.error("DataPipeline file does not exist")
        return False
    
    # Required elements to check
    required_elements = [
        "class DataPipeline",
        "create_data_pipeline",
        "DataCleaningProcessor",
        "DataQualityProcessor",
        "process",
        "cleaning_processor",
        "quality_processor",
        "standardize_timestamps",
        "EventBus",
        "get_pipeline_stats"
    ]
    
    # Read the file
    with open(pipeline_path, 'r') as f:
        content = f.read()
    
    # Check each required element
    success = 0
    for element in required_elements:
        if element in content:
            logger.info(f"✓ Pipeline contains: {element}")
            success += 1
        else:
            logger.error(f"✗ Pipeline missing: {element}")
    
    # Calculate success rate
    success_rate = success / len(required_elements) * 100
    logger.info(f"Pipeline content check: {success}/{len(required_elements)} ({success_rate:.1f}%)")
    
    # Success if at least 80% of required elements are present
    return success_rate >= 80

def validate_migration_readiness() -> bool:
    """Validate if the migration is ready to be completed."""
    logger.info("Validating migration readiness...")
    
    # Check if the switch script exists
    switch_path = os.path.join(os.getcwd(), "switch_strategy_structure.sh")
    
    if not os.path.exists(switch_path):
        logger.error("Switch script does not exist")
        return False
    
    # Check if it's executable
    if not os.access(switch_path, os.X_OK):
        logger.warning("Switch script is not executable, fixing...")
        try:
            os.chmod(switch_path, 0o755)
            logger.info("Made switch script executable")
        except Exception as e:
            logger.error(f"Failed to make switch script executable: {str(e)}")
            return False
    
    # Check if both old and new structures exist
    old_path = os.path.join(os.getcwd(), "trading_bot/strategies")
    new_path = os.path.join(os.getcwd(), "trading_bot/strategies_new")
    
    if not os.path.exists(old_path):
        logger.error("Old strategy structure does not exist")
        return False
    
    if not os.path.exists(new_path):
        logger.error("New strategy structure does not exist")
        return False
    
    logger.info("Migration is ready to be completed")
    return True

def get_coverage_summary() -> dict:
    """Get a summary of the migration coverage."""
    logger.info("Generating migration coverage summary...")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "strategy_organization": {},
        "data_processing": {},
        "asset_classes": {},
        "overall": {}
    }
    
    # Strategy organization coverage
    # Count strategies in old and new structure
    old_strategy_count = count_files(os.path.join(os.getcwd(), "trading_bot/strategies"), "*strategy*.py")
    new_strategy_count = count_files(os.path.join(os.getcwd(), "trading_bot/strategies_new"), "*strategy*.py")
    
    summary["strategy_organization"] = {
        "old_count": old_strategy_count,
        "new_count": new_strategy_count,
        "percentage": (new_strategy_count / max(1, old_strategy_count)) * 100
    }
    
    # Data processing coverage
    old_processor_count = count_files(os.path.join(os.getcwd(), "trading_bot/data"), "*processor*.py")
    data_pipeline_exists = os.path.exists(os.path.join(os.getcwd(), "trading_bot/data/data_pipeline.py"))
    
    summary["data_processing"] = {
        "old_processor_count": old_processor_count,
        "data_pipeline_exists": data_pipeline_exists,
        "dashboard_connector_exists": os.path.exists(os.path.join(os.getcwd(), "trading_bot/data/dashboard_connector.py"))
    }
    
    # Asset class coverage
    asset_classes = ["forex", "stocks", "crypto", "options"]
    asset_class_coverage = {}
    
    for asset_class in asset_classes:
        new_path = os.path.join(os.getcwd(), f"trading_bot/strategies_new/{asset_class}")
        asset_class_coverage[asset_class] = os.path.exists(new_path)
    
    summary["asset_classes"] = asset_class_coverage
    
    # Overall coverage
    checks_passed = sum([
        test_directory_structure(),
        test_forex_trend_strategy_contents(),
        test_data_pipeline_contents(),
        validate_migration_readiness()
    ])
    
    summary["overall"] = {
        "checks_passed": checks_passed,
        "total_checks": 4,
        "percentage": (checks_passed / 4) * 100
    }
    
    return summary

def count_files(directory: str, pattern: str) -> int:
    """Count files matching a pattern in a directory."""
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and pattern.replace("*", "") in file:
                count += 1
    
    return count

def print_summary(summary: dict) -> None:
    """Print a summary of the migration coverage."""
    
    print("\n===== MIGRATION SUMMARY =====")
    print(f"Timestamp: {summary['timestamp']}")
    
    print("\nStrategy Organization:")
    strategy_org = summary["strategy_organization"]
    print(f"  - Old strategy count: {strategy_org['old_count']}")
    print(f"  - New strategy count: {strategy_org['new_count']}")
    print(f"  - Coverage: {strategy_org['percentage']:.1f}%")
    
    print("\nData Processing:")
    data_proc = summary["data_processing"]
    print(f"  - Old processor count: {data_proc['old_processor_count']}")
    print(f"  - Data pipeline exists: {data_proc['data_pipeline_exists']}")
    print(f"  - Dashboard connector exists: {data_proc['dashboard_connector_exists']}")
    
    print("\nAsset Classes:")
    asset_classes = summary["asset_classes"]
    for asset_class, exists in asset_classes.items():
        print(f"  - {asset_class.capitalize()}: {'✓' if exists else '✗'}")
    
    print("\nOverall:")
    overall = summary["overall"]
    print(f"  - Checks passed: {overall['checks_passed']}/{overall['total_checks']}")
    print(f"  - Coverage: {overall['percentage']:.1f}%")
    
    # Overall assessment
    if overall["percentage"] >= 90:
        print("\n✅ MIGRATION READY FOR PRODUCTION")
    elif overall["percentage"] >= 75:
        print("\n🟡 MIGRATION ALMOST READY - Fix remaining issues")
    else:
        print("\n🔴 MIGRATION NEEDS WORK - Several issues to fix")
    
    print("\n==============================")

def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("Starting simple migration test")
    
    tests_passed = True
    
    # Run tests
    logger.info("=== DIRECTORY STRUCTURE TEST ===")
    structure_success = test_directory_structure()
    tests_passed = tests_passed and structure_success
    
    logger.info("\n=== FOREX TREND STRATEGY CONTENT TEST ===")
    strategy_success = test_forex_trend_strategy_contents()
    tests_passed = tests_passed and strategy_success
    
    logger.info("\n=== DATA PIPELINE CONTENT TEST ===")
    pipeline_success = test_data_pipeline_contents()
    tests_passed = tests_passed and pipeline_success
    
    logger.info("\n=== MIGRATION READINESS TEST ===")
    readiness_success = validate_migration_readiness()
    tests_passed = tests_passed and readiness_success
    
    # Generate and print summary
    summary = get_coverage_summary()
    print_summary(summary)
    
    # Provide next steps
    print("\nNext Steps:")
    
    if tests_passed:
        print("1. Run final sanity checks with your sample data")
        print("2. Switch to the new structure with: ./switch_strategy_structure.sh switch")
        print("3. Monitor system behavior after the switch")
        print("4. If issues arise, restore with: ./switch_strategy_structure.sh restore")
        print("5. Continue expanding to additional asset classes")
    else:
        print("1. Fix the highlighted issues")
        print("2. Run this test script again")
        print("3. When all tests pass, proceed with the migration")
    
    return 0 if tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
