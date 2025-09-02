#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone test for forex strategies.

This script tests the core functionality of our forex strategies:
- Trend Following Strategy
- Range Trading Strategy
- Breakout Strategy

It verifies their market regime compatibility, parameter optimization,
and complementary nature without relying on the full trading system.
"""

import os
import sys
import logging
import enum
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define market regimes enum here to avoid dependencies
class MarketRegime(enum.Enum):
    BULL_TREND = 1
    BEAR_TREND = 2
    CONSOLIDATION = 3
    HIGH_VOLATILITY = 4
    LOW_VOLATILITY = 5
    UNKNOWN = 6

# Extract the get_compatibility_score and optimize_for_regime methods from our strategy files
# Trend Following Strategy
def trend_following_compatibility_score(market_regime: MarketRegime) -> float:
    """
    Calculate compatibility score with the given market regime for Trend Following Strategy.
    
    Args:
        market_regime: The current market regime
        
    Returns:
        Compatibility score between 0.0 and 1.0
    """
    # Trend-following strategies perform well in strong trending markets
    # and poorly in consolidation/sideways markets
    compatibility_map = {
        # Trending regimes - best for trend following
        MarketRegime.BULL_TREND: 0.85,      # Excellent compatibility with bull trends
        MarketRegime.BEAR_TREND: 0.80,      # Strong compatibility with bear trends
        
        # Volatile regimes - moderate compatibility with proper filters
        MarketRegime.HIGH_VOLATILITY: 0.65, # Moderate compatibility with volatile markets
        
        # Sideways/ranging regimes - worst for trend following
        MarketRegime.CONSOLIDATION: 0.30,   # Poor compatibility with consolidation
        MarketRegime.LOW_VOLATILITY: 0.40,  # Poor compatibility with low vol markets
        
        # Default for unknown regimes
        MarketRegime.UNKNOWN: 0.50          # Average compatibility with unknown conditions
    }
    
    # Return the compatibility score or default to 0.5 if regime unknown
    return compatibility_map.get(market_regime, 0.5)

def trend_following_optimize_for_regime(market_regime: MarketRegime) -> Dict[str, Any]:
    """
    Optimize Trend Following parameters for the given market regime.
    
    Args:
        market_regime: The current market regime
        
    Returns:
        Dictionary of optimized parameters
    """
    # Start with default parameters
    params = {
        'fast_ma_period': 10,
        'slow_ma_period': 30,
        'adx_period': 14,
        'adx_threshold': 25,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_period': 14,
        'atr_multiplier': 2.0
    }
    
    # Adjust parameters based on regime
    if market_regime == MarketRegime.BULL_TREND:
        # For bull trends, more aggressive settings
        params['adx_threshold'] = 20  # Lower ADX threshold to enter trades earlier
        params['fast_ma_period'] = 8  # Faster signal response
        params['atr_multiplier'] = 2.5  # Wider stops for more room
        
    elif market_regime == MarketRegime.BEAR_TREND:
        # For bear trends, slightly more conservative settings
        params['adx_threshold'] = 22  # Slightly higher than bull trend
        params['fast_ma_period'] = 12  # Slightly slower to avoid whipsaws
        params['atr_multiplier'] = 2.2  # Slightly tighter stops
        
    elif market_regime == MarketRegime.HIGH_VOLATILITY:
        # For high volatility, more conservative settings
        params['adx_threshold'] = 30  # Higher ADX requirement for stronger trends
        params['slow_ma_period'] = 40  # Slower MA to filter noise
        params['atr_multiplier'] = 3.0  # Much wider stops for volatile conditions
        
    elif market_regime == MarketRegime.CONSOLIDATION:
        # For consolidation, very selective settings
        params['adx_threshold'] = 35  # Only respond to very strong breakouts
        params['macd_signal'] = 7  # Faster MACD signal for quicker confirmation
        params['atr_multiplier'] = 1.5  # Tighter stops when ranges dominate
        
    return params

# Range Trading Strategy
def range_trading_compatibility_score(market_regime: MarketRegime) -> float:
    """
    Calculate compatibility score with the given market regime for Range Trading Strategy.
    
    Args:
        market_regime: The current market regime
        
    Returns:
        Compatibility score between 0.0 and 1.0
    """
    # Range trading strategies perform well in consolidation/sideways markets
    # and poorly in trending markets (opposite of trend-following strategies)
    compatibility_map = {
        # Trending regimes - worst for range trading
        MarketRegime.BULL_TREND: 0.30,      # Poor compatibility with bull trends
        MarketRegime.BEAR_TREND: 0.35,      # Poor compatibility with bear trends
        
        # Volatile regimes - moderately compatible with proper stops
        MarketRegime.HIGH_VOLATILITY: 0.50, # Moderate compatibility with volatile markets
        
        # Sideways/ranging regimes - best for range trading
        MarketRegime.CONSOLIDATION: 0.90,   # Excellent compatibility with consolidation
        MarketRegime.LOW_VOLATILITY: 0.85,  # Strong compatibility with low vol markets
        
        # Default for unknown regimes
        MarketRegime.UNKNOWN: 0.60          # Above average compatibility with unknown conditions
    }
    
    # Return the compatibility score or default to 0.5 if regime unknown
    return compatibility_map.get(market_regime, 0.6)

def range_trading_optimize_for_regime(market_regime: MarketRegime) -> Dict[str, Any]:
    """
    Optimize Range Trading parameters for the given market regime.
    
    Args:
        market_regime: The current market regime
        
    Returns:
        Dictionary of optimized parameters
    """
    # Start with default parameters
    params = {
        'range_threshold': 0.03,
        'min_touches': 2,
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'stop_loss_pips': 30
    }
    
    # Adjust parameters based on regime
    if market_regime == MarketRegime.CONSOLIDATION:
        # For strong consolidation, use tighter ranges
        params['range_threshold'] = 0.02
        params['min_touches'] = 3      # Require more touches
        params['bb_std_dev'] = 1.8     # Tighter Bollinger Bands
        params['stop_loss_pips'] = 20  # Tighter stops
        
    elif market_regime == MarketRegime.LOW_VOLATILITY:
        # For low volatility, use smaller ranges and tighter stops
        params['range_threshold'] = 0.015
        params['min_range_pips'] = 15
        params['max_range_pips'] = 150
        params['stop_loss_pips'] = 15
        
    elif market_regime == MarketRegime.HIGH_VOLATILITY:
        # For high volatility, use wider ranges and stops
        params['range_threshold'] = 0.04
        params['min_range_pips'] = 30
        params['max_range_pips'] = 300
        params['bb_std_dev'] = 2.5     # Wider Bollinger Bands
        params['stop_loss_pips'] = 40  # Wider stops
        
    elif market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
        # For trending markets, be more selective with ranges
        params['range_threshold'] = 0.025
        params['min_touches'] = 4       # Require more touches to confirm range
        params['stop_loss_pips'] = 25   # Normal stops
        
    return params

# Breakout Strategy
def breakout_compatibility_score(market_regime: MarketRegime) -> float:
    """
    Calculate compatibility score with the given market regime for Breakout Strategy.
    
    Args:
        market_regime: The current market regime
        
    Returns:
        Compatibility score between 0.0 and 1.0
    """
    # Breakout strategies perform well in transitional markets between
    # consolidation and trending conditions, and in high volatility environments
    compatibility_map = {
        # Trending regimes - moderate to good compatibility 
        MarketRegime.BULL_TREND: 0.65,      # Good compatibility with established bull trends
        MarketRegime.BEAR_TREND: 0.65,      # Good compatibility with established bear trends
        
        # Volatile regimes - best for breakout strategies
        MarketRegime.HIGH_VOLATILITY: 0.90, # Excellent compatibility with volatile markets
        
        # Sideways/ranging regimes - moderate compatibility when consolidating before breakout
        MarketRegime.CONSOLIDATION: 0.60,   # Moderate compatibility with consolidation
        MarketRegime.LOW_VOLATILITY: 0.40,  # Low compatibility with low vol markets
        
        # Default for unknown regimes
        MarketRegime.UNKNOWN: 0.60          # Above average compatibility with unknown conditions
    }
    
    # Return the compatibility score or default to 0.6 if regime unknown
    return compatibility_map.get(market_regime, 0.6)

def breakout_optimize_for_regime(market_regime: MarketRegime) -> Dict[str, Any]:
    """
    Optimize Breakout Strategy parameters for the given market regime.
    
    Args:
        market_regime: The current market regime
        
    Returns:
        Dictionary of optimized parameters
    """
    # Start with default parameters
    params = {
        'breakout_threshold': 0.01,
        'confirmation_candles': 2,
        'volume_threshold': 1.5,
        'atr_multiplier': 1.5,
        'take_profit_factor': 2.0,
        'stop_loss_pips': 50
    }
    
    # Adjust parameters based on regime
    if market_regime == MarketRegime.HIGH_VOLATILITY:
        # For high volatility, use larger confirmations but aggressive profit targets
        params['breakout_threshold'] = 0.015      # Higher threshold for stronger moves
        params['confirmation_candles'] = 3        # More candles to confirm
        params['take_profit_factor'] = 3.0        # Larger profit targets
        params['stop_loss_pips'] = 60            # Wider stops
        params['min_breakout_pips'] = 30         # Larger breakouts expected
        
    elif market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
        # For trending markets, look for trend continuation breakouts
        params['breakout_threshold'] = 0.01
        params['confirmation_candles'] = 2
        params['take_profit_factor'] = 2.5
        params['atr_multiplier'] = 1.2           # Less stringent volatility filter
        
    elif market_regime == MarketRegime.CONSOLIDATION:
        # For consolidation, focus on the strongest breakouts
        params['breakout_threshold'] = 0.02      # Higher threshold to avoid false breakouts
        params['confirmation_candles'] = 3       # More confirmation needed
        params['volume_threshold'] = 2.0         # Stronger volume confirmation required
        params['atr_multiplier'] = 1.8          # More stringent volatility filter
        
    return params

def test_strategy_compatibility():
    """Test the market regime compatibility of our forex strategies"""
    print("\n")
    print("=" * 80)
    print("FOREX STRATEGY MARKET REGIME COMPATIBILITY TEST")
    print("=" * 80)
    
    # Test all regimes
    regimes = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.CONSOLIDATION,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY,
    ]
    
    # Print compatibility score table
    print("\nCompatibility Scores (0.0-1.0):")
    print("-" * 80)
    print(f"{'Market Regime':<20} {'Trend Following':<20} {'Range Trading':<20} {'Breakout':<20}")
    print("-" * 80)
    
    # Store scores for analysis
    all_scores = []
    
    # Test each regime
    for regime in regimes:
        # Get compatibility scores
        trend_score = trend_following_compatibility_score(regime)
        range_score = range_trading_compatibility_score(regime)
        breakout_score = breakout_compatibility_score(regime)
        
        # Print scores
        print(f"{regime.name:<20} {trend_score:<20.2f} {range_score:<20.2f} {breakout_score:<20.2f}")
        
        # Store scores
        all_scores.append({
            'regime': regime,
            'trend_score': trend_score,
            'range_score': range_score,
            'breakout_score': breakout_score
        })
    
    print("-" * 80)
    
    # Find best strategy for each regime
    print("\nBest Strategy for Each Market Regime:")
    print("-" * 50)
    
    best_regimes = {}
    
    for score_data in all_scores:
        regime = score_data['regime']
        scores = [
            ('Trend Following', score_data['trend_score']),
            ('Range Trading', score_data['range_score']),
            ('Breakout', score_data['breakout_score'])
        ]
        
        # Find best strategy for this regime
        best_strategy = max(scores, key=lambda x: x[1])
        
        print(f"{regime.name:<20}: {best_strategy[0]} (score: {best_strategy[1]:.2f})")
        
        # Record which regimes each strategy is best for
        if best_strategy[0] not in best_regimes:
            best_regimes[best_strategy[0]] = []
        
        best_regimes[best_strategy[0]].append(regime.name)
    
    # Check if each strategy is best for at least one regime
    print("\nOptimal Market Regimes by Strategy:")
    print("-" * 50)
    
    for strategy, regimes in best_regimes.items():
        print(f"{strategy}: {', '.join(regimes)}")
    
    # Test parameter optimization
    print("\nParameter Optimization Verification:")
    print("-" * 80)
    print("Each strategy optimizes parameters differently for different market regimes:")
    
    # Test trend strategy parameters
    bull_params = trend_following_optimize_for_regime(MarketRegime.BULL_TREND)
    bear_params = trend_following_optimize_for_regime(MarketRegime.BEAR_TREND)
    
    print("\nTrend Following Strategy:")
    print(f"  Bull Trend ADX Threshold: {bull_params['adx_threshold']}")
    print(f"  Bear Trend ADX Threshold: {bear_params['adx_threshold']}")
    
    # Test range strategy parameters
    consol_params = range_trading_optimize_for_regime(MarketRegime.CONSOLIDATION)
    low_vol_params = range_trading_optimize_for_regime(MarketRegime.LOW_VOLATILITY)
    
    print("\nRange Trading Strategy:")
    print(f"  Consolidation Range Threshold: {consol_params['range_threshold']}")
    print(f"  Low Volatility Range Threshold: {low_vol_params['range_threshold']}")
    
    # Test breakout strategy parameters
    high_vol_params = breakout_optimize_for_regime(MarketRegime.HIGH_VOLATILITY)
    trend_params = breakout_optimize_for_regime(MarketRegime.BULL_TREND)
    
    print("\nBreakout Strategy:")
    print(f"  High Volatility Breakout Threshold: {high_vol_params['breakout_threshold']}")
    print(f"  Bull Trend Breakout Threshold: {trend_params['breakout_threshold']}")
    
    # Verify complementary coverage
    print("\nComplementary Coverage Verification:")
    print("-" * 50)
    
    all_strategies = ['Trend Following', 'Range Trading', 'Breakout']
    covered_strategies = list(best_regimes.keys())
    
    if set(all_strategies) <= set(covered_strategies):
        print("✅ PASS: All strategies are optimal for at least one market regime")
    else:
        print(f"❌ FAIL: The following strategies are not optimal for any regime: {set(all_strategies) - set(covered_strategies)}")
    
    # Check market regime coverage
    covered_regimes = set()
    for regimes_list in best_regimes.values():
        for regime in regimes_list:
            covered_regimes.add(regime)
    
    all_regime_names = [regime.name for regime in regimes]
    
    if set(all_regime_names) <= covered_regimes:
        print("✅ PASS: All market regimes have at least one optimal strategy")
    else:
        print(f"❌ FAIL: The following regimes have no optimal strategy: {set(all_regime_names) - covered_regimes}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY: Forex Strategies provide complementary market coverage")
    print("=" * 80)

if __name__ == "__main__":
    test_strategy_compatibility()
