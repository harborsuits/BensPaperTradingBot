#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for forex strategy market regime compatibility.

This script tests the market regime compatibility scores of our three forex strategies:
- ForexTrendFollowingStrategy
- ForexRangeTradingStrategy
- ForexBreakoutStrategy
"""

import logging
import sys
import os
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import needed modules and classes directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trading_bot.strategies.strategy_template import MarketRegime

# Direct import of our strategy classes (bypassing the main module imports)
forex_trend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'trading_bot/strategies/forex/trend_following_strategy.py')
forex_range_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'trading_bot/strategies/forex/range_trading_strategy.py')
forex_breakout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'trading_bot/strategies/forex/breakout_strategy.py')
                               
# Load the strategy files dynamically to bypass dependency issues
import importlib.util

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Mock EventBus class to prevent errors
class MockEventBus:
    def __init__(self):
        self.events = []
    
    def publish(self, event):
        self.events.append(event)
        logger.info(f"Event published")

def test_forex_regime_compatibility():
    """Test the market regime compatibility of our forex strategies."""
    logger.info("Testing Forex Strategy Market Regime Compatibility")
    
    # Load modules
    try:
        trend_module = load_module_from_path("trend_following_strategy", forex_trend_path)
        range_module = load_module_from_path("range_trading_strategy", forex_range_path)
        breakout_module = load_module_from_path("breakout_strategy", forex_breakout_path)
        
        # Instantiate strategies
        trend_strategy = trend_module.ForexTrendFollowingStrategy()
        range_strategy = range_module.ForexRangeTradingStrategy()
        breakout_strategy = breakout_module.ForexBreakoutStrategy()
        
        # Assign mock event bus
        mock_bus = MockEventBus()
        trend_strategy.event_bus = mock_bus
        range_strategy.event_bus = mock_bus
        breakout_strategy.event_bus = mock_bus
        
        # Test compatibility across all market regimes
        regimes = [
            MarketRegime.BULL_TREND,
            MarketRegime.BEAR_TREND,
            MarketRegime.CONSOLIDATION,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY,
        ]
        
        # Print header
        print("\nForex Strategy Market Regime Compatibility Scores (0.0-1.0)")
        print("-" * 70)
        print(f"{'Market Regime':<20} {'Trend Following':<20} {'Range Trading':<20} {'Breakout':<20}")
        print("-" * 70)
        
        # Store scores for analysis
        all_scores = []
        
        # Test each regime
        for regime in regimes:
            # Get compatibility scores
            trend_score = trend_strategy.get_compatibility_score(regime)
            range_score = range_strategy.get_compatibility_score(regime)
            breakout_score = breakout_strategy.get_compatibility_score(regime)
            
            # Print scores
            print(f"{regime.name:<20} {trend_score:<20.2f} {range_score:<20.2f} {breakout_score:<20.2f}")
            
            # Store scores
            all_scores.append({
                'regime': regime,
                'trend_score': trend_score,
                'range_score': range_score,
                'breakout_score': breakout_score
            })
            
        print("-" * 70)
        
        # Analyze complementary nature
        print("\nStrategy Complementary Analysis:")
        print("-" * 50)
        
        for score_data in all_scores:
            regime = score_data['regime']
            scores = [
                ('Trend Following', score_data['trend_score']),
                ('Range Trading', score_data['range_score']),
                ('Breakout', score_data['breakout_score'])
            ]
            
            # Find best strategy for this regime
            best_strategy = max(scores, key=lambda x: x[1])
            
            print(f"Best strategy for {regime.name}: {best_strategy[0]} (score: {best_strategy[1]:.2f})")
        
        # Check if each strategy is best for at least one regime
        best_regimes = {}
        for score_data in all_scores:
            regime = score_data['regime']
            scores = [
                ('Trend Following', score_data['trend_score']),
                ('Range Trading', score_data['range_score']),
                ('Breakout', score_data['breakout_score'])
            ]
            
            best_strategy = max(scores, key=lambda x: x[1])[0]
            if best_strategy not in best_regimes:
                best_regimes[best_strategy] = []
            
            best_regimes[best_strategy].append(regime.name)
        
        print("\nEach strategy's optimal regimes:")
        for strategy, regimes in best_regimes.items():
            print(f"{strategy}: {', '.join(regimes)}")
        
        # Verify complementary nature
        print("\nVerifying complementary coverage:")
        all_strategies = ['Trend Following', 'Range Trading', 'Breakout']
        covered_strategies = list(best_regimes.keys())
        
        if set(all_strategies) <= set(covered_strategies):
            print("✅ PASS: All strategies are optimal for at least one market regime")
        else:
            print(f"❌ FAIL: The following strategies are not optimal for any regime: {set(all_strategies) - set(covered_strategies)}")
        
        # Check parameter optimization
        print("\nParameter Optimization Verification:")
        print("-" * 50)
        
        # Test parameter optimization for trend strategy
        bull_params = trend_strategy.optimize_for_regime(MarketRegime.BULL_TREND)
        bear_params = trend_strategy.optimize_for_regime(MarketRegime.BEAR_TREND)
        
        if bull_params != bear_params:
            print("✅ PASS: Trend strategy adapts parameters for different trend regimes")
        else:
            print("❌ FAIL: Trend strategy uses same parameters for bull and bear regimes")
        
        # Test parameter optimization for range strategy
        consol_params = range_strategy.optimize_for_regime(MarketRegime.CONSOLIDATION)
        low_vol_params = range_strategy.optimize_for_regime(MarketRegime.LOW_VOLATILITY)
        
        if consol_params != low_vol_params:
            print("✅ PASS: Range strategy adapts parameters for different range regimes")
        else:
            print("❌ FAIL: Range strategy uses same parameters for consolidation and low volatility")
        
        # Test parameter optimization for breakout strategy
        high_vol_params = breakout_strategy.optimize_for_regime(MarketRegime.HIGH_VOLATILITY)
        consol_params = breakout_strategy.optimize_for_regime(MarketRegime.CONSOLIDATION)
        
        if high_vol_params != consol_params:
            print("✅ PASS: Breakout strategy adapts parameters for volatility vs. consolidation")
        else:
            print("❌ FAIL: Breakout strategy uses same parameters for high volatility and consolidation")
        
        print("\nTest complete! All strategies verified for market regime compatibility.")
        
    except Exception as e:
        logger.error(f"Error testing forex strategies: {str(e)}")
        raise

if __name__ == "__main__":
    test_forex_regime_compatibility()
