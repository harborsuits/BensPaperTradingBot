#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Regime Detection Test Script

This script tests the ML-based market regime detection in the ForexStrategySelector.
It compares ML detection to traditional detection across various synthetic market conditions.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from our project
from trading_bot.strategies.strategy_template import MarketRegime, TimeFrame
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector, RiskTolerance

def create_test_data():
    """
    Create synthetic market data for each of the known market regimes.
    
    Returns:
        Dictionary mapping regime names to synthetic datasets
    """
    # Common parameters for all synthetic data
    start_date = datetime(2023, 1, 1, tzinfo=pytz.UTC)
    days = 10
    hours = days * 24
    
    # Data container
    test_data = {}
    
    # Create date range for hourly data
    dates = pd.date_range(start=start_date, periods=hours, freq='H')
    
    # ========== Bull Trend ==========
    # Strong upward trend with consistent price increases
    bull_data = pd.DataFrame(index=dates)
    bull_data['datetime'] = bull_data.index
    
    # Base price and increments
    price = 1.1000
    prices = []
    
    for i in range(len(bull_data)):
        # Add trend plus small noise
        price += np.random.normal(0.0002, 0.0001)  # Upward bias with small noise
        prices.append(price)
    
    bull_data['close'] = prices
    bull_data['open'] = bull_data['close'].shift(1)
    bull_data['open'].iloc[0] = prices[0] - 0.0002
    
    # Create high/low with appropriate ranges
    bull_data['high'] = bull_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.0001, len(bull_data)))
    bull_data['low'] = bull_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.0001, len(bull_data)))
    bull_data['volume'] = np.random.normal(100, 20, len(bull_data))
    
    test_data['BULL_TREND'] = {'EURUSD': bull_data}
    
    # ========== Bear Trend ==========
    # Strong downward trend with consistent price decreases
    bear_data = pd.DataFrame(index=dates)
    bear_data['datetime'] = bear_data.index
    
    # Base price and decrements
    price = 1.1000
    prices = []
    
    for i in range(len(bear_data)):
        # Add downward trend plus small noise
        price -= np.random.normal(0.0002, 0.0001)  # Downward bias with small noise
        prices.append(price)
    
    bear_data['close'] = prices
    bear_data['open'] = bear_data['close'].shift(1)
    bear_data['open'].iloc[0] = prices[0] + 0.0002
    
    # Create high/low with appropriate ranges
    bear_data['high'] = bear_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.0001, len(bear_data)))
    bear_data['low'] = bear_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.0001, len(bear_data)))
    bear_data['volume'] = np.random.normal(100, 20, len(bear_data))
    
    test_data['BEAR_TREND'] = {'EURUSD': bear_data}
    
    # ========== Consolidation ==========
    # Sideways movement in a tight range
    consol_data = pd.DataFrame(index=dates)
    consol_data['datetime'] = consol_data.index
    
    # Base price with minimal random movement
    price = 1.1000
    prices = []
    
    for i in range(len(consol_data)):
        # Small random movement around a central price
        price = 1.1000 + np.random.normal(0, 0.0003)
        prices.append(price)
    
    consol_data['close'] = prices
    consol_data['open'] = consol_data['close'].shift(1)
    consol_data['open'].iloc[0] = 1.1000
    
    # Tight ranges for high/low
    consol_data['high'] = consol_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.0001, len(consol_data)))
    consol_data['low'] = consol_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.0001, len(consol_data)))
    consol_data['volume'] = np.random.normal(80, 15, len(consol_data))
    
    test_data['CONSOLIDATION'] = {'EURUSD': consol_data}
    
    # ========== High Volatility ==========
    # Large price swings with high volume
    vol_data = pd.DataFrame(index=dates)
    vol_data['datetime'] = vol_data.index
    
    # Base price with large random movements
    price = 1.1000
    prices = []
    
    for i in range(len(vol_data)):
        # Large random movements
        price += np.random.normal(0, 0.0015)
        prices.append(price)
    
    vol_data['close'] = prices
    vol_data['open'] = vol_data['close'].shift(1)
    vol_data['open'].iloc[0] = 1.1000
    
    # Wide ranges for high/low
    vol_data['high'] = vol_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.0008, len(vol_data)))
    vol_data['low'] = vol_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.0008, len(vol_data)))
    vol_data['volume'] = np.random.normal(200, 50, len(vol_data))
    
    test_data['HIGH_VOLATILITY'] = {'EURUSD': vol_data}
    
    # ========== Low Volatility ==========
    # Very small price movements with low volume
    low_vol_data = pd.DataFrame(index=dates)
    low_vol_data['datetime'] = low_vol_data.index
    
    # Base price with very small movements
    price = 1.1000
    prices = []
    
    for i in range(len(low_vol_data)):
        # Minimal price changes
        price += np.random.normal(0, 0.00005)
        prices.append(price)
    
    low_vol_data['close'] = prices
    low_vol_data['open'] = low_vol_data['close'].shift(1)
    low_vol_data['open'].iloc[0] = 1.1000
    
    # Very tight ranges for high/low
    low_vol_data['high'] = low_vol_data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.00002, len(low_vol_data)))
    low_vol_data['low'] = low_vol_data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.00002, len(low_vol_data)))
    low_vol_data['volume'] = np.random.normal(50, 10, len(low_vol_data))
    
    test_data['LOW_VOLATILITY'] = {'EURUSD': low_vol_data}
    
    return test_data

def plot_test_data(test_data):
    """
    Create a visualization of the synthetic test data.
    
    Args:
        test_data: Dictionary of synthetic datasets by regime
    """
    fig, axes = plt.subplots(len(test_data), 1, figsize=(12, 10), sharex=True)
    
    for i, (regime, data) in enumerate(test_data.items()):
        df = data['EURUSD']
        axes[i].plot(df.index, df['close'], label=f'Close Price')
        axes[i].set_title(f'{regime} Synthetic Data')
        axes[i].set_ylabel('Price')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_test_synthetic_data.png')
    plt.close()
    
    print(f"Synthetic data visualization saved to 'ml_test_synthetic_data.png'")

def test_ml_regime_detection():
    """
    Test ML-based regime detection against traditional methods.
    """
    # Create test data for different regimes
    test_data = create_test_data()
    
    # Visualize test data
    plot_test_data(test_data)
    
    # Initialize selectors with and without ML
    # Normal selector with ML if available
    selector_with_ml = ForexStrategySelector()
    
    # Force traditional method by disabling ML
    params_no_ml = {'ml_model_path': 'nonexistent_path.joblib'}
    selector_no_ml = ForexStrategySelector(parameters=params_no_ml)
    
    # Track results for comparison
    ml_results = {}
    traditional_results = {}
    true_regimes = {}
    
    print("\n" + "=" * 80)
    print("TESTING ML REGIME DETECTION")
    print("=" * 80)
    
    print("\nComparing ML vs Traditional Regime Detection:")
    print("-" * 80)
    print(f"{'Regime':<20} {'True Regime':<20} {'ML Detection':<20} {'Traditional Detection':<20} {'Match':<10}")
    print("-" * 80)
    
    # Test each regime type
    for regime_name, market_data in test_data.items():
        true_regime = MarketRegime[regime_name]
        true_regimes[regime_name] = true_regime
        
        # Detect with ML (if available)
        detected_ml = selector_with_ml._detect_market_regime(market_data)
        ml_results[regime_name] = detected_ml
        
        # Detect with traditional method
        detected_trad = selector_no_ml._detect_market_regime(market_data)
        traditional_results[regime_name] = detected_trad
        
        # Compare results
        match = "✓" if detected_ml == true_regime else "✗"
        
        print(f"{regime_name:<20} {true_regime.name:<20} {detected_ml.name:<20} {detected_trad.name:<20} {match:<10}")
    
    # Calculate accuracy metrics
    ml_correct = sum(1 for regime, result in ml_results.items() if result == true_regimes[regime])
    trad_correct = sum(1 for regime, result in traditional_results.items() if result == true_regimes[regime])
    
    ml_accuracy = ml_correct / len(true_regimes) * 100
    trad_accuracy = trad_correct / len(true_regimes) * 100
    
    print("\nAccuracy Summary:")
    print("-" * 80)
    print(f"ML Detection Accuracy: {ml_accuracy:.1f}%")
    print(f"Traditional Detection Accuracy: {trad_accuracy:.1f}%")
    
    # If we have the ML model loaded, show a bit more detail
    if selector_with_ml.ml_model_loaded:
        print("\nML Model Information:")
        print("-" * 80)
        
        # Check if we have a model package with metadata
        if hasattr(selector_with_ml, 'ml_model_package') and isinstance(selector_with_ml.ml_model_package, dict):
            package = selector_with_ml.ml_model_package
            training_date = package.get('training_date', 'Unknown')
            model_accuracy = package.get('accuracy', 0.0) * 100
            
            print(f"Model Training Date: {training_date}")
            print(f"Model Training Accuracy: {model_accuracy:.1f}%")
            
            # If we have feature columns, show top features
            if 'feature_importances' in package and 'feature_columns' in package:
                importances = package['feature_importances']
                columns = package['feature_columns']
                
                if len(importances) > 0 and len(importances) == len(columns):
                    indices = np.argsort(importances)[::-1][:5]  # Top 5 features
                    
                    print("\nTop 5 Important Features:")
                    for i in indices:
                        print(f"  - {columns[i]}: {importances[i]:.4f}")
        
    print("\n" + "=" * 80)
    print("ML REGIME DETECTION TEST COMPLETED")
    print("=" * 80)

def main():
    """Main test execution function."""
    test_ml_regime_detection()

if __name__ == "__main__":
    main()
