#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update All Strategies

This script updates all strategy implementations to follow the established best practices
from our Iron Condor implementation. It ensures consistent:
- Error handling
- Event processing
- Validation logic
- Integration with the strategy adjustment system

Run this script to update all strategies at once.
"""

import os
import logging
import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, project_root)

from trading_bot.strategies_new.validation.strategy_validator import StrategyValidator
from trading_bot.strategies_new.validation.strategy_updater import StrategyUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(script_dir, 'strategy_update.log'))
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Update all strategy implementations to follow best practices')
    parser.add_argument('--dry-run', action='store_true', help='Validate strategies without making changes')
    parser.add_argument('--strategy-type', type=str, choices=['options', 'stocks', 'forex', 'crypto', 'all'],
                      default='all', help='Type of strategies to update')
    parser.add_argument('--specific-strategy', type=str, help='Specific strategy file to update')
    args = parser.parse_args()
    
    strategies_dir = os.path.join(project_root, 'trading_bot', 'strategies_new')
    
    # Set the target directory based on strategy type
    if args.strategy_type != 'all':
        strategies_dir = os.path.join(strategies_dir, args.strategy_type)
    
    logger.info(f"Starting strategy update process for {args.strategy_type} strategies")
    logger.info(f"Target directory: {strategies_dir}")
    
    # Handle specific strategy case
    if args.specific_strategy:
        if not os.path.exists(args.specific_strategy):
            # Try to find it relative to the strategies directory
            potential_path = os.path.join(strategies_dir, args.specific_strategy)
            if os.path.exists(potential_path):
                specific_strategy_path = potential_path
            else:
                logger.error(f"Strategy file not found: {args.specific_strategy}")
                return
        else:
            specific_strategy_path = args.specific_strategy
            
        logger.info(f"Processing specific strategy: {specific_strategy_path}")
        
        if args.dry_run:
            # Just validate the strategy
            # We need to import the module and get the strategy class
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("strategy_module", specific_strategy_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find the strategy class
                strategy_class = None
                for name in dir(module):
                    if "Strategy" in name and not name.startswith("__"):
                        strategy_class = getattr(module, name)
                        break
                
                if strategy_class:
                    result = StrategyValidator.validate_strategy_class(strategy_class)
                    _print_validation_result(specific_strategy_path, result)
                else:
                    logger.error(f"No strategy class found in {specific_strategy_path}")
            except Exception as e:
                logger.error(f"Error validating {specific_strategy_path}: {str(e)}")
        else:
            # Update the strategy
            result = StrategyUpdater.update_strategy(specific_strategy_path)
            _print_update_result(specific_strategy_path, result)
    else:
        # Process all strategies in the directory
        if args.dry_run:
            logger.info("Performing dry run (validation only)")
            results = StrategyValidator.validate_all_strategies(strategies_dir)
            
            # Print summary
            valid_count = sum(1 for result in results.values() if result["is_valid"])
            invalid_count = len(results) - valid_count
            
            logger.info(f"Validation complete: {valid_count} valid strategies, {invalid_count} need updates")
            
            # Print details for invalid strategies
            if invalid_count > 0:
                logger.info("\nStrategies needing updates:")
                for name, result in results.items():
                    if not result["is_valid"]:
                        _print_validation_result(name, result)
        else:
            logger.info("Updating all strategies to follow best practices")
            results = StrategyUpdater.update_all_strategies(strategies_dir)
            
            # Print summary
            success_count = sum(1 for result in results.values() if not result["errors"])
            error_count = len(results) - success_count
            
            logger.info(f"Update complete: {success_count} strategies updated successfully, {error_count} had errors")
            
            # Print details for strategies with errors
            if error_count > 0:
                logger.info("\nStrategies with errors:")
                for path, result in results.items():
                    if result["errors"]:
                        _print_update_result(path, result)
    
    logger.info("Strategy update process complete")

def _print_validation_result(name, result):
    """Print a validation result in a readable format."""
    logger.info(f"\n{name}:")
    
    if result["missing_methods"]:
        logger.info("  Missing methods:")
        for method in result["missing_methods"]:
            logger.info(f"    - {method}")
    
    if result["methods_missing_error_handling"]:
        logger.info("  Methods missing error handling:")
        for method in result["methods_missing_error_handling"]:
            logger.info(f"    - {method}")
    
    if result["missing_event_subscriptions"]:
        logger.info("  Missing event subscriptions:")
        for event in result["missing_event_subscriptions"]:
            logger.info(f"    - {event}")
    
    if result["missing_validation_checks"]:
        logger.info("  Missing validation checks:")
        for check in result["missing_validation_checks"]:
            logger.info(f"    - {check}")

def _print_update_result(path, result):
    """Print an update result in a readable format."""
    logger.info(f"\n{path}:")
    
    if result["updates_applied"]:
        logger.info("  Updates applied:")
        for update in result["updates_applied"]:
            logger.info(f"    - {update}")
    
    if result["updates_skipped"]:
        logger.info("  Updates skipped:")
        for update in result["updates_skipped"]:
            logger.info(f"    - {update}")
    
    if result["errors"]:
        logger.info("  Errors:")
        for error in result["errors"]:
            logger.info(f"    - {error}")

if __name__ == "__main__":
    main()
