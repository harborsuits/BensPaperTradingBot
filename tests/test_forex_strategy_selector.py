#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Forex Strategy Selector.

This script tests the intelligent strategy selection based on:
1. Market conditions (regime detection)
2. Time-awareness (trading sessions)
3. Risk tolerance parameters
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from enum import Enum

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the MarketRegime enum from a local definition if necessary
from trading_bot.strategies.strategy_template import MarketRegime, TimeFrame
from trading_bot.strategies.base.forex_base import ForexSession
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector, RiskTolerance

def create_test_data():
    """
    Create synthetic market data for testing different regimes.
    
    Returns:
        Dictionary with synthetic data for different market regimes.
    """
    # Date range for our synthetic data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
    
    # Create base dataframe with dates
    base_df = pd.DataFrame(index=dates)
    base_df['datetime'] = base_df.index
    
    # Generate synthetic data for different regimes
    data_sets = {}
    
    # --------------- Bull Trend Data ---------------
    bull_trend = base_df.copy()
    # Start price and daily increments
    price = 1.2000
    daily_increment = 0.0010
    hourly_noise = 0.0002
    
    prices = []
    for i in range(len(bull_trend)):
        # Add a small trend component plus noise
        noise = np.random.normal(0, hourly_noise)
        price += (daily_increment / 24) + noise
        prices.append(price)
    
    bull_trend['close'] = prices
    # Generate OHLC data
    bull_trend['open'] = bull_trend['close'].shift(1)
    bull_trend['open'].iloc[0] = prices[0] - daily_increment / 48
    bull_trend['high'] = bull_trend[['open', 'close']].max(axis=1) + np.random.uniform(0, hourly_noise, len(bull_trend))
    bull_trend['low'] = bull_trend[['open', 'close']].min(axis=1) - np.random.uniform(0, hourly_noise, len(bull_trend))
    # Add volume
    bull_trend['volume'] = np.random.uniform(100, 200, len(bull_trend))
    
    data_sets['BULL_TREND'] = {'EURUSD': bull_trend}
    
    # --------------- Bear Trend Data ---------------
    bear_trend = base_df.copy()
    # Start price and daily decrements
    price = 1.2000
    daily_decrement = 0.0010
    hourly_noise = 0.0002
    
    prices = []
    for i in range(len(bear_trend)):
        # Add a small trend component plus noise
        noise = np.random.normal(0, hourly_noise)
        price -= (daily_decrement / 24) + noise
        prices.append(price)
    
    bear_trend['close'] = prices
    bear_trend['open'] = bear_trend['close'].shift(1)
    bear_trend['open'].iloc[0] = prices[0] + daily_decrement / 48
    bear_trend['high'] = bear_trend[['open', 'close']].max(axis=1) + np.random.uniform(0, hourly_noise, len(bear_trend))
    bear_trend['low'] = bear_trend[['open', 'close']].min(axis=1) - np.random.uniform(0, hourly_noise, len(bear_trend))
    bear_trend['volume'] = np.random.uniform(100, 200, len(bear_trend))
    
    data_sets['BEAR_TREND'] = {'EURUSD': bear_trend}
    
    # --------------- Consolidation Data ---------------
    consolidation = base_df.copy()
    # Start price and very small noise around a central value
    price = 1.2000
    hourly_noise = 0.0003
    
    prices = []
    for i in range(len(consolidation)):
        # Just add noise around the central value
        noise = np.random.normal(0, hourly_noise)
        price += noise
        prices.append(price)
    
    consolidation['close'] = prices
    consolidation['open'] = consolidation['close'].shift(1)
    consolidation['open'].iloc[0] = prices[0]
    consolidation['high'] = consolidation[['open', 'close']].max(axis=1) + np.random.uniform(0, hourly_noise, len(consolidation))
    consolidation['low'] = consolidation[['open', 'close']].min(axis=1) - np.random.uniform(0, hourly_noise, len(consolidation))
    consolidation['volume'] = np.random.uniform(50, 100, len(consolidation))  # Lower volume in consolidation
    
    data_sets['CONSOLIDATION'] = {'EURUSD': consolidation}
    
    # --------------- High Volatility Data ---------------
    high_vol = base_df.copy()
    # Start price and very high noise
    price = 1.2000
    hourly_noise = 0.0008
    
    prices = []
    for i in range(len(high_vol)):
        # Large noise component
        noise = np.random.normal(0, hourly_noise)
        price += noise
        prices.append(price)
    
    high_vol['close'] = prices
    high_vol['open'] = high_vol['close'].shift(1)
    high_vol['open'].iloc[0] = prices[0]
    high_vol['high'] = high_vol[['open', 'close']].max(axis=1) + np.random.uniform(0, hourly_noise*2, len(high_vol))
    high_vol['low'] = high_vol[['open', 'close']].min(axis=1) - np.random.uniform(0, hourly_noise*2, len(high_vol))
    high_vol['volume'] = np.random.uniform(200, 300, len(high_vol))  # Higher volume in volatile markets
    
    data_sets['HIGH_VOLATILITY'] = {'EURUSD': high_vol}
    
    # --------------- Low Volatility Data ---------------
    low_vol = base_df.copy()
    # Start price and very low noise
    price = 1.2000
    hourly_noise = 0.0001
    
    prices = []
    for i in range(len(low_vol)):
        # Tiny noise component
        noise = np.random.normal(0, hourly_noise)
        price += noise
        prices.append(price)
    
    low_vol['close'] = prices
    low_vol['open'] = low_vol['close'].shift(1)
    low_vol['open'].iloc[0] = prices[0]
    low_vol['high'] = low_vol[['open', 'close']].max(axis=1) + np.random.uniform(0, hourly_noise, len(low_vol))
    low_vol['low'] = low_vol[['open', 'close']].min(axis=1) - np.random.uniform(0, hourly_noise, len(low_vol))
    low_vol['volume'] = np.random.uniform(30, 80, len(low_vol))  # Lower volume in low volatility
    
    data_sets['LOW_VOLATILITY'] = {'EURUSD': low_vol}
    
    return data_sets

def test_regime_detection():
    """Test automatic regime detection with synthetic data."""
    print("\n" + "=" * 80)
    print("TESTING REGIME DETECTION")
    print("=" * 80)
    
    # Create test data for different regimes
    data_sets = create_test_data()
    
    # Initialize strategy selector with medium risk
    selector = ForexStrategySelector(risk_tolerance=RiskTolerance.MEDIUM)
    
    # Test regime detection with each dataset
    for regime_name, market_data in data_sets.items():
        detected_regime = selector._detect_market_regime(market_data)
        expected_regime = MarketRegime[regime_name]
        
        result = "✓ PASS" if detected_regime == expected_regime else f"✗ FAIL (detected {detected_regime.name})"
        print(f"Expected: {regime_name:<15} | Detected: {detected_regime.name:<15} | Result: {result}")

def test_session_detection():
    """Test forex session detection based on time."""
    print("\n" + "=" * 80)
    print("TESTING FOREX SESSION DETECTION")
    print("=" * 80)
    
    # Initialize strategy selector
    selector = ForexStrategySelector(time_zone="UTC")
    
    # Test different times (in UTC)
    test_times = [
        # Format: (datetime, expected_sessions, description)
        (datetime(2023, 5, 1, 5, 0, tzinfo=pytz.UTC), 
         [ForexSession.SYDNEY, ForexSession.TOKYO], 
         "Sydney-Tokyo sessions"),
        
        (datetime(2023, 5, 1, 8, 30, tzinfo=pytz.UTC), 
         [ForexSession.TOKYO, ForexSession.LONDON], 
         "Tokyo-London overlap"),
        
        (datetime(2023, 5, 1, 14, 0, tzinfo=pytz.UTC), 
         [ForexSession.LONDON, ForexSession.NEWYORK, ForexSession.LONDON_NEWYORK_OVERLAP], 
         "London-NY overlap (highest volatility)"),
        
        (datetime(2023, 5, 1, 20, 0, tzinfo=pytz.UTC), 
         [ForexSession.NEWYORK], 
         "NY session only"),
        
        (datetime(2023, 5, 1, 23, 0, tzinfo=pytz.UTC), 
         [ForexSession.SYDNEY], 
         "Sydney session start"),
        
        (datetime(2023, 5, 6, 12, 0, tzinfo=pytz.UTC),  # Saturday
         [], 
         "Weekend (market closed)")
    ]
    
    for test_time, expected_sessions, description in test_times:
        detected_sessions = selector._identify_active_sessions(test_time)
        
        detected_names = [s.name for s in detected_sessions]
        expected_names = [s.name for s in expected_sessions]
        
        result = "✓ PASS" if set(detected_names) == set(expected_names) else "✗ FAIL"
        
        print(f"Time: {test_time.strftime('%Y-%m-%d %H:%M')} UTC | {description}")
        print(f"Expected: {expected_names}")
        print(f"Detected: {detected_names}")
        print(f"Result: {result}\n")

def test_strategy_selection():
    """Test strategy selection across various conditions."""
    print("\n" + "=" * 80)
    print("TESTING STRATEGY SELECTION")
    print("=" * 80)
    
    # Create test data for different regimes
    data_sets = create_test_data()
    
    # Test parameters
    risk_levels = [RiskTolerance.LOW, RiskTolerance.MEDIUM, RiskTolerance.HIGH]
    time_samples = [
        datetime(2023, 5, 1, 5, 0, tzinfo=pytz.UTC),     # Sydney-Tokyo
        datetime(2023, 5, 1, 14, 0, tzinfo=pytz.UTC),    # London-NY overlap
        datetime(2023, 5, 1, 20, 0, tzinfo=pytz.UTC)     # NY only
    ]
    
    test_configs = []
    for risk in risk_levels:
        for time_sample in time_samples:
            for regime_name, market_data in data_sets.items():
                test_configs.append((risk, time_sample, regime_name, market_data))
    
    # Run tests
    results = []
    for risk, time_sample, regime_name, market_data in test_configs:
        # Initialize selector with this risk level
        selector = ForexStrategySelector(risk_tolerance=risk)
        
        # Force the regime rather than detecting it (for testing purposes)
        regime = MarketRegime[regime_name]
        
        # Select optimal strategy
        strategy_name, params = selector.select_optimal_strategy(
            market_data=market_data,
            current_time=time_sample,
            detected_regime=regime
        )
        
        active_sessions = [s.name for s in selector._identify_active_sessions(time_sample)]
        
        results.append({
            'Risk': risk,
            'Time': time_sample.strftime('%H:%M UTC'),
            'Sessions': ", ".join(active_sessions) if active_sessions else "None (weekend)",
            'Regime': regime_name,
            'Selected Strategy': strategy_name,
            'Risk Parameters': {
                'Stop Loss Mult': params.get('stop_loss_atr_mult', 'N/A'),
                'Take Profit Mult': params.get('take_profit_atr_mult', 'N/A')
            }
        })
    
    # Print formatted results
    print("\nStrategy Selection Results:")
    print("-" * 100)
    headers = ['Risk', 'Time', 'Sessions', 'Regime', 'Selected Strategy', 'Risk Parameters']
    print(f"{headers[0]:<10} {headers[1]:<12} {headers[2]:<25} {headers[3]:<16} {headers[4]:<20} {headers[5]:<25}")
    print("-" * 100)
    
    for r in results:
        risk_params = f"SL:{r['Risk Parameters']['Stop Loss Mult']:.2f}, TP:{r['Risk Parameters']['Take Profit Mult']:.2f}"
        print(f"{r['Risk']:<10} {r['Time']:<12} {r['Sessions']:<25} {r['Regime']:<16} {r['Selected Strategy']:<20} {risk_params:<25}")

def test_risk_parameters():
    """Test how risk parameters are adjusted based on risk tolerance."""
    print("\n" + "=" * 80)
    print("TESTING RISK PARAMETER ADJUSTMENTS")
    print("=" * 80)
    
    risk_levels = [RiskTolerance.LOW, RiskTolerance.MEDIUM, RiskTolerance.HIGH]
    
    print("\nRisk Parameter Comparison:")
    print("-" * 80)
    
    # Print table header
    print(f"{'Parameter':<25} {'Low Risk':<15} {'Medium Risk':<15} {'High Risk':<15}")
    print("-" * 80)
    
    risk_param_results = {}
    for risk in risk_levels:
        selector = ForexStrategySelector(risk_tolerance=risk)
        risk_params = selector.get_risk_parameters()
        
        # Store for comparison
        risk_key = 'Low Risk' if risk == RiskTolerance.LOW else 'Medium Risk' if risk == RiskTolerance.MEDIUM else 'High Risk'
        risk_param_results[risk_key] = risk_params
    
    # Print parameters for comparison
    for param in ['position_size_multiplier', 'stop_loss_multiplier', 'take_profit_multiplier', 
                 'max_trades_per_session', 'max_risk_per_trade_pct', 'max_risk_per_day_pct']:
        low_val = risk_param_results['Low Risk'][param]
        med_val = risk_param_results['Medium Risk'][param]
        high_val = risk_param_results['High Risk'][param]
        
        print(f"{param:<25} {low_val:<15} {med_val:<15} {high_val:<15}")

def main():
    print("\n" + "=" * 80)
    print("FOREX STRATEGY SELECTOR TEST SUITE")
    print("=" * 80)
    print("\nTesting intelligent selection of forex strategies based on:")
    print("1. Market conditions (regime detection)")
    print("2. Time-awareness (trading sessions)")
    print("3. Risk tolerance parameters")
    
    # Run test suite
    test_regime_detection()
    test_session_detection()
    test_strategy_selection()
    test_risk_parameters()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
