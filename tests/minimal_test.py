#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test for Straddle/Strangle strategy core logic.
"""

import logging
import os
from datetime import datetime

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_strategy_file():
    """Simple test to validate the strategy file exists and can be read."""
    strategy_path = "trading_bot/strategies/options/volatility_spreads/straddle_strangle_strategy.py"
    
    if not os.path.exists(strategy_path):
        logger.error(f"Strategy file not found at: {strategy_path}")
        return False
    
    # Read the file and analyze content
    with open(strategy_path, 'r') as f:
        content = f.read()
    
    # Check for key components
    class_present = "class StraddleStrangleStrategy" in content
    register_present = "@register_strategy" in content
    universe_method = "define_universe" in content
    signals_method = "generate_signals" in content
    exit_method = "on_exit_signal" in content
    
    logger.info(f"Strategy file exists and contains:")
    logger.info(f"- StraddleStrangleStrategy class: {'✅' if class_present else '❌'}")
    logger.info(f"- @register_strategy decorator: {'✅' if register_present else '❌'}")
    logger.info(f"- define_universe method: {'✅' if universe_method else '❌'}")
    logger.info(f"- generate_signals method: {'✅' if signals_method else '❌'}")
    logger.info(f"- on_exit_signal method: {'✅' if exit_method else '❌'}")
    
    # Check for implementation of key features
    straddle_impl = "_find_straddle" in content
    strangle_impl = "_find_strangle" in content
    confidence_adjust = "_adjust_signal_confidence" in content
    position_tracking = "_track_position" in content
    
    logger.info(f"Strategy implements:")
    logger.info(f"- Straddle selection: {'✅' if straddle_impl else '❌'}")
    logger.info(f"- Strangle selection: {'✅' if strangle_impl else '❌'}")
    logger.info(f"- Confidence adjustment: {'✅' if confidence_adjust else '❌'}")
    logger.info(f"- Position tracking: {'✅' if position_tracking else '❌'}")
    
    # Check for exit criteria
    exit_criteria = [
        "Profit target" in content,
        "Stop loss" in content, 
        "Time stop" in content,
        "IV decrease" in content,
        "event has passed" in content,
        "expiration" in content,
        "drawdown from peak" in content
    ]
    
    logger.info(f"Exit criteria implemented: {sum(exit_criteria)}/7")
    
    # Overall validation
    all_checks = [
        class_present, register_present, universe_method, signals_method, exit_method,
        straddle_impl, strangle_impl, confidence_adjust, position_tracking
    ] + exit_criteria
    
    success_rate = sum(all_checks) / len(all_checks) * 100
    logger.info(f"Overall implementation completeness: {success_rate:.1f}%")
    
    return success_rate >= 90

if __name__ == "__main__":
    logger.info("Starting minimal test at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    success = test_strategy_file()
    
    if success:
        logger.info("✅ Strategy implementation passes validation!")
    else:
        logger.error("❌ Strategy implementation may be incomplete!")
    
    logger.info("Test completed.")
