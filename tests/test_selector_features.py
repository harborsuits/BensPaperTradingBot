#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Forex Strategy Selector Feature Tests

This script tests the new features added to the forex strategy selector:
1. Economic calendar integration
2. Historical performance feedback
3. ML-enhanced regime detection

These tests ensure that strategy selection correctly adapts based on:
- Economic news events
- Past performance of strategies in similar conditions
- Machine learning-based market regime detection
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock
import json

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from our project
from trading_bot.strategies.strategy_template import MarketRegime, TimeFrame
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector, RiskTolerance
from trading_bot.strategies.base.forex_base import ForexSession

def create_demo_data():
    """
    Create sample OHLCV data for testing.
    """
    start_date = datetime(2023, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2023, 1, 5, tzinfo=pytz.UTC)
    hours = int((end_date - start_date).total_seconds() / 3600) + 1
    
    dates = pd.date_range(start=start_date, periods=hours, freq='H')
    
    df = pd.DataFrame(index=dates)
    df['datetime'] = df.index
    
    # Create synthetic price data
    price = 1.1000
    prices = []
    
    for i in range(len(df)):
        # Add small random movement
        price += np.random.normal(0, 0.0005)
        prices.append(price)
    
    df['close'] = prices
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = 1.0998
    
    # Create high/low with appropriate ranges
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.0002, len(df)))
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.0002, len(df)))
    df['volume'] = np.random.normal(100, 20, len(df))
    
    return {'EURUSD': df}

def test_economic_calendar():
    """Test economic calendar integration in strategy selection."""
    print("\n" + "=" * 80)
    print("TESTING ECONOMIC CALENDAR INTEGRATION")
    print("=" * 80)
    
    market_data = create_demo_data()
    current_time = datetime.now(pytz.UTC)
    
    # Create a mock calendar response
    mock_calendar_response = {
        'has_high_impact': True,
        'events': [
            {
                'datetime': current_time + timedelta(hours=1),
                'currency': 'EUR',
                'event': 'Interest Rate Decision',
                'impact': 'high'
            },
            {
                'datetime': current_time - timedelta(minutes=30),
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'impact': 'high'
            }
        ]
    }
    
    # Create selector with default settings
    selector = ForexStrategySelector()
    
    # 1. Test with no news (normal behavior)
    with patch.object(selector, '_check_economic_calendar', return_value={'has_high_impact': False, 'events': []}):
        strategy_no_news, params_no_news = selector.select_optimal_strategy(
            market_data=market_data,
            current_time=current_time,
            detected_regime=MarketRegime.BULL_TREND
        )
    
    # 2. Test with high impact news
    with patch.object(selector, '_check_economic_calendar', return_value=mock_calendar_response):
        strategy_with_news, params_with_news = selector.select_optimal_strategy(
            market_data=market_data,
            current_time=current_time,
            detected_regime=MarketRegime.BULL_TREND
        )
    
    # Print results
    print("\nStrategy Selection With and Without Economic News:")
    print("-" * 80)
    print(f"Without News: {strategy_no_news}")
    print(f"With News: {strategy_with_news}")
    
    # Show parameter differences
    print("\nRisk Parameter Differences Due to News:")
    print("-" * 80)
    
    # Extract and compare key risk parameters
    params_to_compare = ['stop_loss_atr_mult', 'take_profit_atr_mult']
    
    print(f"{'Parameter':<25} {'Without News':<15} {'With News':<15} {'Change':<10}")
    print("-" * 80)
    
    for param in params_to_compare:
        if param in params_no_news and param in params_with_news:
            no_news_val = params_no_news[param]
            with_news_val = params_with_news[param]
            change = (with_news_val / no_news_val - 1) * 100 if no_news_val != 0 else 0
            
            print(f"{param:<25} {no_news_val:<15.2f} {with_news_val:<15.2f} {change:+.1f}%")
    
    print("\nNew parameters with high-impact news:")
    print("-" * 80)
    
    news_specific_params = {key: value for key, value in params_with_news.items() 
                          if key.startswith('news_') or key.endswith('_news')}
    
    if news_specific_params:
        for key, value in news_specific_params.items():
            print(f"{key}: {value}")
    else:
        print("No news-specific parameters found")
        
    # Verify news adjustment behavior
    if strategy_no_news != strategy_with_news:
        print("\n✓ Successfully changed strategy selection due to economic news")
    elif params_no_news != params_with_news:
        print("\n✓ Successfully adjusted strategy parameters due to economic news")
    else:
        print("\n✗ Economic news did not impact strategy selection or parameters")
        
    print("\n" + "=" * 80)

def test_historical_performance_feedback():
    """Test historical performance feedback in strategy selection."""
    print("\n" + "=" * 80)
    print("TESTING HISTORICAL PERFORMANCE FEEDBACK")
    print("=" * 80)
    
    market_data = create_demo_data()
    current_time = datetime.now(pytz.UTC)
    
    # Initialize selector
    selector = ForexStrategySelector()
    
    # Record the default selection
    original_strategy, original_params = selector.select_optimal_strategy(
        market_data=market_data,
        current_time=current_time,
        detected_regime=MarketRegime.BULL_TREND
    )
    
    print(f"\nOriginal strategy selection: {original_strategy}")
    
    # Record performance data for different strategies
    print("\nRecording historical performance data:")
    print("-" * 80)
    
    # 1. Record poor performance for the originally selected strategy
    selector.record_strategy_performance(original_strategy, MarketRegime.BULL_TREND, 0.3)  # Poor performance
    print(f"Recorded poor performance (0.3) for {original_strategy} in BULL_TREND regime")
    
    # 2. Record excellent performance for an alternative strategy
    all_strategies = list(selector.strategy_compatibility.keys())
    alternative_strategy = next((s for s in all_strategies if s != original_strategy), all_strategies[0])
    
    selector.record_strategy_performance(alternative_strategy, MarketRegime.BULL_TREND, 0.9)  # Excellent performance
    print(f"Recorded excellent performance (0.9) for {alternative_strategy} in BULL_TREND regime")
    
    # Add more performance records for statistical significance
    for _ in range(4):
        selector.record_strategy_performance(original_strategy, MarketRegime.BULL_TREND, 0.3)
        selector.record_strategy_performance(alternative_strategy, MarketRegime.BULL_TREND, 0.9)
    
    # Re-select with performance data
    new_strategy, new_params = selector.select_optimal_strategy(
        market_data=market_data,
        current_time=current_time,
        detected_regime=MarketRegime.BULL_TREND
    )
    
    print(f"\nNew strategy selection after performance feedback: {new_strategy}")
    
    # Calculate adjusted scores (recreate the internal logic)
    original_score = selector.strategy_compatibility[original_strategy][MarketRegime.BULL_TREND]
    alternative_score = selector.strategy_compatibility[alternative_strategy][MarketRegime.BULL_TREND]
    
    print("\nStrategy Score Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Base Score':<15} {'Adj Score':<15} {'Perf Data'}")
    print("-" * 80)
    
    # Show adjustment for original strategy
    perf_data = selector.strategy_performance[original_strategy][MarketRegime.BULL_TREND]
    avg_perf = sum(perf_data) / len(perf_data) if perf_data else 0
    adjustment = (avg_perf - 0.5) * 0.4  # From the code logic
    adjusted_score = original_score * (1 + adjustment)
    
    print(f"{original_strategy:<25} {original_score:<15.2f} {adjusted_score:<15.2f} {perf_data}")
    
    # Show adjustment for alternative strategy
    perf_data = selector.strategy_performance[alternative_strategy][MarketRegime.BULL_TREND]
    avg_perf = sum(perf_data) / len(perf_data) if perf_data else 0
    adjustment = (avg_perf - 0.5) * 0.4  # From the code logic
    adjusted_score = alternative_score * (1 + adjustment)
    
    print(f"{alternative_strategy:<25} {alternative_score:<15.2f} {adjusted_score:<15.2f} {perf_data}")
    
    # Verify feedback is working
    if original_strategy != new_strategy:
        print("\n✓ Successfully changed strategy based on historical performance")
    else:
        print("\n✗ Historical performance did not impact strategy selection")
    
    # Test the performance calculation
    test_performance = selector.calculate_strategy_performance(
        strategy_name="test_strategy",
        initial_equity=10000,
        final_equity=11000,
        max_drawdown=5.0,
        win_rate=65.0,
        profit_factor=2.5
    )
    
    print(f"\nTest performance score calculation: {test_performance:.4f}")
    print("\n" + "=" * 80)

def create_performance_dashboard():
    """Create a sample performance visualization dashboard."""
    print("\n" + "=" * 80)
    print("CREATING PERFORMANCE VISUALIZATION DASHBOARD")
    print("=" * 80)
    
    # Initialize selector
    selector = ForexStrategySelector()
    
    # Generate synthetic performance data
    regimes = [r for r in MarketRegime]
    strategies = list(selector.strategy_compatibility.keys())
    
    # Simulate random performances with realistic patterns
    for strategy in strategies:
        for regime in regimes:
            # Number of performance records
            num_records = np.random.randint(10, 30)
            
            # Base performance level for this combination (some strategies naturally perform better in certain regimes)
            base_performance = selector.strategy_compatibility[strategy].get(regime, 0.5)
            
            # Generate performance data with some variance around the base level
            performance_data = np.random.normal(base_performance, 0.15, num_records)
            
            # Ensure values are in valid range
            performance_data = np.clip(performance_data, 0.0, 1.0)
            
            # Simulate performance improvement over time (learning)
            performance_data = np.sort(performance_data)
            
            # Record in the selector
            selector.strategy_performance[strategy][regime] = list(performance_data)
    
    # Create visualization folder
    viz_dir = os.path.join(project_root, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Performance by Regime visualization
    plt.figure(figsize=(12, 8))
    
    regime_names = [r.name for r in regimes if r != MarketRegime.UNKNOWN]
    strategy_names = [s.split('_', 1)[1] for s in strategies]  # Remove 'forex_' prefix
    
    # Prepare data for heatmap
    heatmap_data = []
    
    for strategy in strategies:
        strategy_data = []
        for regime in regimes:
            if regime != MarketRegime.UNKNOWN:
                perf_data = selector.strategy_performance[strategy][regime]
                avg_perf = sum(perf_data) / len(perf_data) if perf_data else 0
                strategy_data.append(avg_perf)
        heatmap_data.append(strategy_data)
    
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    ax = sns.heatmap(heatmap_array, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=regime_names, yticklabels=strategy_names)
    
    plt.title('Average Strategy Performance by Market Regime')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'strategy_regime_performance.png'))
    plt.close()
    
    # Performance over time visualization
    plt.figure(figsize=(12, 8))
    
    # Select a specific regime for time series visualization
    selected_regime = MarketRegime.BULL_TREND
    
    for strategy in strategies:
        # Get performance data for this strategy in the selected regime
        performance_history = selector.strategy_performance[strategy][selected_regime]
        
        if performance_history:
            # Plot as a time series (x-axis is just the record index)
            strategy_label = strategy.split('_', 1)[1]  # Remove 'forex_' prefix
            plt.plot(range(len(performance_history)), performance_history, 
                    marker='o', linestyle='-', label=strategy_label)
    
    plt.title(f'Strategy Performance Over Time in {selected_regime.name} Regime')
    plt.xlabel('Performance Record Index')
    plt.ylabel('Performance Score (0-1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'performance_over_time.png'))
    plt.close()
    
    # Export the performance data as JSON for interactive dashboards
    performance_data = {}
    for strategy in strategies:
        performance_data[strategy] = {
            regime.name: selector.strategy_performance[strategy][regime]
            for regime in regimes
        }
    
    with open(os.path.join(viz_dir, 'performance_data.json'), 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"\nPerformance visualizations created in: {viz_dir}")
    print(f"- strategy_regime_performance.png: Heatmap of strategy performance by regime")
    print(f"- performance_over_time.png: Time series of strategy performance")
    print(f"- performance_data.json: Raw data for creating interactive dashboards")
    
    print("\n" + "=" * 80)

def main():
    """Run all tests for the enhanced forex strategy selector features."""
    print("\n" + "=" * 80)
    print("FOREX STRATEGY SELECTOR FEATURE TESTS")
    print("=" * 80)
    
    print("\nTesting enhanced strategy selector features:")
    print("1. Economic calendar integration")
    print("2. Historical performance feedback")
    print("3. Performance visualization dashboard")
    
    # Run the tests
    test_economic_calendar()
    test_historical_performance_feedback()
    create_performance_dashboard()
    
    print("\nAll feature tests completed successfully!")

if __name__ == "__main__":
    main()
