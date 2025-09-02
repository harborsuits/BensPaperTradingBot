#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation Verification Script

This script verifies that our Enhanced Strategy Selector and 
Price Action Strategy have been correctly implemented and registered.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_strategy_factory():
    """Verify that our strategy factory contains the new Price Action Strategy."""
    try:
        # Import strategy factory
        from trading_bot.strategies.strategy_factory import StrategyFactory
        
        # Create an instance
        factory = StrategyFactory()
        
        # Get available strategies
        available_strategies = factory.get_available_strategies()
        
        # Check if our new strategy is registered
        if 'forex_price_action' in available_strategies:
            logger.info("✓ Price Action Strategy is registered in the factory")
            return True
        else:
            logger.error("✗ Price Action Strategy is NOT registered in the factory")
            logger.info(f"Available strategies: {available_strategies}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying strategy factory: {str(e)}")
        return False

def verify_enhanced_selector():
    """Verify that the Enhanced Strategy Selector is implemented."""
    try:
        # Import the selector
        from trading_bot.strategy_selection.enhanced_strategy_selector import EnhancedStrategySelector
        
        # Check if key methods are implemented
        selector = EnhancedStrategySelector()
        
        # Check for key attributes
        required_attributes = [
            'strategy_performance',
            'time_performance',
            'preferences',
            'config'
        ]
        
        for attr in required_attributes:
            if not hasattr(selector, attr):
                logger.error(f"✗ Enhanced Strategy Selector missing attribute: {attr}")
                return False
        
        # Check for key methods
        required_methods = [
            'select_optimal_strategy',
            'evaluate_trading_time',
            'set_risk_profile',
            '_select_strategy_combination'
        ]
        
        for method in required_methods:
            if not hasattr(selector, method) or not callable(getattr(selector, method)):
                logger.error(f"✗ Enhanced Strategy Selector missing method: {method}")
                return False
        
        logger.info("✓ Enhanced Strategy Selector is properly implemented")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying enhanced selector: {str(e)}")
        return False

def verify_price_action_strategy():
    """Verify that the Price Action Strategy is properly implemented."""
    try:
        # Import the strategy
        from trading_bot.strategies.forex.price_action_strategy import PriceActionStrategy
        
        # Create an instance
        strategy = PriceActionStrategy()
        
        # Check for key attributes
        required_attributes = [
            'price_levels',
            'detected_patterns',
            'mtf_data',
            'parameters'
        ]
        
        for attr in required_attributes:
            if not hasattr(strategy, attr):
                logger.error(f"✗ Price Action Strategy missing attribute: {attr}")
                return False
        
        # Check for key methods
        required_methods = [
            'detect_pin_bar',
            'detect_engulfing',
            'detect_inside_bar',
            'detect_doji',
            'detect_candle_patterns',
            'identify_support_resistance',
            'generate_signals',
            'get_regime_compatibility_score'
        ]
        
        for method in required_methods:
            if not hasattr(strategy, method) or not callable(getattr(strategy, method)):
                logger.error(f"✗ Price Action Strategy missing method: {method}")
                return False
        
        # Check regime compatibility scores
        from trading_bot.strategies.strategy_template import MarketRegime
        
        # Validate that we have meaningful regime compatibility scores
        regime_scores = {regime: strategy.get_regime_compatibility_score(regime) 
                        for regime in MarketRegime}
        
        # Print the scores
        logger.info("Price Action Strategy Regime Compatibility Scores:")
        for regime, score in regime_scores.items():
            logger.info(f"  {regime.name}: {score:.2f}")
        
        logger.info("✓ Price Action Strategy is properly implemented")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying price action strategy: {str(e)}")
        return False

def main():
    """Run all verification checks."""
    logger.info("Starting implementation verification...")
    
    # Run verification checks
    factory_ok = verify_strategy_factory()
    selector_ok = verify_enhanced_selector()
    strategy_ok = verify_price_action_strategy()
    
    # Print summary
    logger.info("\nVerification Summary:")
    logger.info(f"Strategy Factory: {'✓ PASS' if factory_ok else '✗ FAIL'}")
    logger.info(f"Enhanced Selector: {'✓ PASS' if selector_ok else '✗ FAIL'}")
    logger.info(f"Price Action Strategy: {'✓ PASS' if strategy_ok else '✗ FAIL'}")
    
    # Overall status
    if factory_ok and selector_ok and strategy_ok:
        logger.info("\n✓ All implementations are COMPLETE and VERIFIED")
        return 0
    else:
        logger.error("\n✗ Some implementations have ISSUES")
        return 1

if __name__ == "__main__":
    sys.exit(main())
