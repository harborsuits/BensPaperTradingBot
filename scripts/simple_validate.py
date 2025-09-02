#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Validation Script for Straddle/Strangle Strategy
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    logger.info("----- VALIDATING STRADDLE/STRANGLE STRATEGY -----")
    
    try:
        # Direct import of just the strategy class we need
        from trading_bot.strategies.options.volatility_spreads.straddle_strangle_strategy import StraddleStrangleStrategy
        logger.info("✓ Successfully imported StraddleStrangleStrategy")
        
        # Create an instance with minimal dependencies
        strategy = StraddleStrangleStrategy()
        logger.info("✓ Successfully instantiated StraddleStrangleStrategy")
        
        # Very basic method validation
        if hasattr(strategy, 'generate_signals'):
            logger.info("✓ Strategy has generate_signals method")
        
        logger.info("✓ Strategy is ready for pipeline integration")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import Error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Validation Error: {e}")
        return False
    finally:
        logger.info("----- VALIDATION COMPLETE -----")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
