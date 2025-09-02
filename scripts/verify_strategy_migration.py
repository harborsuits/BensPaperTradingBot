#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Strategy Migration

This script verifies that the ForexTrendFollowingStrategy has been properly
migrated to the new structure and can be instantiated and used correctly.
"""

import os
import sys
import logging
import inspect
import importlib.util
from typing import Dict, Any, Optional, Type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerifyMigration")

def verify_imports():
    """Verify that all necessary modules can be imported."""
    logger.info("Verifying imports...")
    
    try:
        # Try importing from the new structure
        import_path = "trading_bot.strategies_new.forex.trend.trend_following_strategy"
        logger.info(f"Attempting to import {import_path}")
        
        # Dynamically import the module
        spec = importlib.util.find_spec(import_path)
        if spec is None:
            logger.error(f"Module {import_path} not found")
            return False
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[import_path] = module
        spec.loader.exec_module(module)
        
        # Check if the class exists
        if not hasattr(module, "ForexTrendFollowingStrategy"):
            logger.error("ForexTrendFollowingStrategy class not found in module")
            return False
        
        strategy_class = getattr(module, "ForexTrendFollowingStrategy")
        logger.info(f"Successfully imported {strategy_class.__name__}")
        
        # Check inheritance
        bases = [base.__name__ for base in inspect.getmro(strategy_class)]
        logger.info(f"Inheritance chain: {' -> '.join(bases)}")
        
        if "ForexBaseStrategy" not in bases:
            logger.error("ForexTrendFollowingStrategy does not inherit from ForexBaseStrategy")
            return False
        
        logger.info("Import verification successful")
        return strategy_class
    
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during import verification: {str(e)}")
        return False

def verify_registration(strategy_class):
    """Verify that the strategy is properly registered."""
    logger.info("Verifying strategy registration...")
    
    try:
        # Import the registry
        from trading_bot.strategies_new.factory.strategy_registry import StrategyRegistry
        
        # Check if the strategy is registered
        all_strategies = StrategyRegistry.get_all_strategy_names()
        strategy_name = strategy_class.__name__
        
        if strategy_name not in all_strategies:
            logger.error(f"Strategy {strategy_name} not found in registry")
            return False
        
        # Check categorizations
        forex_strategies = StrategyRegistry.get_strategies_by_asset_class("forex")
        if strategy_name not in forex_strategies:
            logger.error(f"Strategy {strategy_name} not categorized as forex")
            return False
        
        trend_strategies = StrategyRegistry.get_strategies_by_type("trend_following")
        if strategy_name not in trend_strategies:
            logger.error(f"Strategy {strategy_name} not categorized as trend_following")
            return False
        
        logger.info("Registration verification successful")
        return True
    
    except ImportError as e:
        logger.error(f"Import error during registration verification: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during registration verification: {str(e)}")
        return False

def verify_instantiation(strategy_class):
    """Verify that the strategy can be instantiated."""
    logger.info("Verifying strategy instantiation...")
    
    try:
        # Create an instance
        strategy = strategy_class(
            name="TestForexTrendFollowing",
            parameters={
                'fast_ma_period': 10,
                'slow_ma_period': 20,
                'adx_period': 14
            }
        )
        
        # Check methods and attributes
        required_methods = [
            'calculate_indicators', 
            'generate_signals', 
            'register_events'
        ]
        
        for method in required_methods:
            if not hasattr(strategy, method) or not callable(getattr(strategy, method)):
                logger.error(f"Required method {method} not found or not callable")
                return False
        
        logger.info("Instantiation verification successful")
        return True
    
    except Exception as e:
        logger.error(f"Error during instantiation: {str(e)}")
        return False

def prepare_for_production_switch():
    """Prepare for switching from strategies_new to strategies."""
    logger.info("Preparing for production switch...")
    
    # Create backup commands
    backup_cmd = 'mv /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies_backup'
    switch_cmd = 'mv /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies_new /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies'
    
    logger.info("To switch to the new structure, run the following commands:")
    logger.info(f"1. {backup_cmd}")
    logger.info(f"2. {switch_cmd}")
    
    logger.info("To restore the old structure if needed, run:")
    logger.info('mv /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies_new')
    logger.info('mv /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies_backup /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/strategies')
    
    return {
        "backup_cmd": backup_cmd,
        "switch_cmd": switch_cmd
    }

def main():
    """Main entry point."""
    logger.info("Starting migration verification")
    
    # Verify the imports
    strategy_class = verify_imports()
    if not strategy_class:
        logger.error("Import verification failed")
        return False
    
    # Verify registration
    if not verify_registration(strategy_class):
        logger.error("Registration verification failed")
        return False
    
    # Verify instantiation
    if not verify_instantiation(strategy_class):
        logger.error("Instantiation verification failed")
        return False
    
    # All verifications passed
    logger.info("All verifications passed!")
    logger.info("The ForexTrendFollowingStrategy has been successfully migrated")
    
    # Prepare for production switch
    commands = prepare_for_production_switch()
    
    # Create a switch script
    with open('switch_to_new_strategies.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Switch to new strategy structure\n\n')
        f.write(f'{commands["backup_cmd"]}\n')
        f.write(f'{commands["switch_cmd"]}\n')
        f.write('\necho "Successfully switched to new strategy structure"\n')
    
    logger.info("Created switch_to_new_strategies.sh script")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
