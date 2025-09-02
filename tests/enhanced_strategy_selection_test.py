#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Strategy Selection Test

This script demonstrates the enhanced strategy selection framework
with risk profile integration and trading time optimization.
"""

import os
import sys
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.strategy_selection.enhanced_strategy_selector import EnhancedStrategySelector
from trading_bot.strategy_selection.risk_profile_manager import RiskProfileManager, RiskToleranceLevel
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector
from trading_bot.strategies.strategy_template import MarketRegime, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load or generate sample market data for testing.
    
    Returns:
        Dictionary of OHLCV DataFrames
    """
    # Check if we have saved sample data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')
    os.makedirs(data_path, exist_ok=True)
    
    sample_file = os.path.join(data_path, 'forex_sample_data.pkl')
    
    if os.path.exists(sample_file):
        logger.info(f"Loading sample data from {sample_file}")
        try:
            return pd.read_pickle(sample_file)
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
    
    # Generate synthetic data for different regimes
    logger.info("Generating synthetic forex data")
    
    # Common parameters
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    periods = 500
    
    # Generate data for each symbol
    data = {}
    for symbol in symbols:
        # Create timestamps
        now = pd.Timestamp.now()
        dates = pd.date_range(end=now, periods=periods, freq='1H')
        
        # Initialize price at a reasonable level for each currency pair
        if symbol == 'EURUSD':
            base_price = 1.1000
        elif symbol == 'GBPUSD':
            base_price = 1.2500
        elif symbol == 'USDJPY':
            base_price = 110.00
        elif symbol == 'AUDUSD':
            base_price = 0.7500
        else:
            base_price = 1.0000
        
        # Generate random walk price with some trend and volatility
        np.random.seed(42 + hash(symbol) % 1000)  # Different seed for each symbol
        
        # Create trending segment for the first part
        trend_factor = np.random.choice([-1, 1]) * 0.0001  # Small trend
        trend = np.cumsum(np.random.normal(trend_factor, 0.0020, periods//3))
        
        # Create ranging segment for the middle part
        range_center = trend[-1]
        range_data = np.random.normal(0, 0.0015, periods//3)
        
        # Create volatile segment for the last part
        volatility = np.random.normal(0, 0.0035, periods - 2*(periods//3))
        
        # Combine segments
        random_walk = np.concatenate([trend, range_center + range_data, range_center + volatility])
        
        # Create price series
        close = base_price + np.cumsum(random_walk)
        high = close + np.random.uniform(0.0005, 0.0020, periods)
        low = close - np.random.uniform(0.0005, 0.0020, periods)
        open_price = low + np.random.uniform(0, 1, periods) * (high - low)
        
        # Add volume
        volume = np.random.uniform(100, 1000, periods)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        data[symbol] = df
    
    # Save data for future use
    try:
        pd.to_pickle(data, sample_file)
        logger.info(f"Saved sample data to {sample_file}")
    except Exception as e:
        logger.error(f"Error saving sample data: {str(e)}")
    
    return data

def create_test_risk_profiles() -> List[str]:
    """
    Create test risk profiles for different risk tolerance levels.
    
    Returns:
        List of created profile names
    """
    profile_manager = RiskProfileManager()
    
    # Create profiles for all risk levels
    profiles = []
    
    # Conservative profile
    conservative = profile_manager.create_profile(
        name="conservative_trader",
        tolerance_level=RiskToleranceLevel.CONSERVATIVE,
        timeframe_preferences=[TimeFrame.HOURS_4.name, TimeFrame.DAILY.name],
        preferred_strategy_types=["trend_following", "position", "carry"],
        favorite_currency_pairs=["EURUSD", "USDJPY"],
        custom_settings={
            "max_drawdown_percent": 8.0,
            "diversification_preference": 0.8,
            "symbol_position_adjustments": {"USDJPY": 0.8}  # 20% reduced position size for JPY
        }
    )
    profiles.append("conservative_trader")
    
    # Moderate profile
    moderate = profile_manager.create_profile(
        name="moderate_trader",
        tolerance_level=RiskToleranceLevel.MODERATE,
        timeframe_preferences=[TimeFrame.HOURS_1.name, TimeFrame.HOURS_4.name],
        preferred_strategy_types=["swing", "breakout", "momentum"],
        favorite_currency_pairs=["EURUSD", "GBPUSD", "AUDUSD"],
        custom_settings={
            "max_drawdown_percent": 15.0,
            "diversification_preference": 0.5
        }
    )
    profiles.append("moderate_trader")
    
    # Aggressive profile
    aggressive = profile_manager.create_profile(
        name="aggressive_trader",
        tolerance_level=RiskToleranceLevel.VERY_AGGRESSIVE,
        timeframe_preferences=[TimeFrame.MINUTES_15.name, TimeFrame.HOURS_1.name],
        preferred_strategy_types=["scalping", "day_trading", "momentum"],
        favorite_currency_pairs=["GBPUSD", "AUDUSD"],
        custom_settings={
            "max_drawdown_percent": 30.0,
            "diversification_preference": 0.2,
            "max_open_trades": 10
        }
    )
    profiles.append("aggressive_trader")
    
    return profiles

def test_strategy_selection_with_profiles(data: Dict[str, pd.DataFrame]) -> None:
    """
    Test strategy selection with different risk profiles and market regimes.
    
    Args:
        data: Dictionary of OHLCV DataFrames
    """
    logger.info("Testing strategy selection with different risk profiles")
    
    # Create risk profiles
    profile_names = create_test_risk_profiles()
    
    # Initialize base and enhanced selectors
    base_selector = ForexStrategySelector()
    risk_manager = RiskProfileManager()
    enhanced_selector = EnhancedStrategySelector(base_selector=base_selector, risk_profile_manager=risk_manager)
    
    # Test with each profile
    for profile_name in profile_names:
        logger.info(f"\nTesting with risk profile: {profile_name}")
        
        # Load profile
        enhanced_selector.set_risk_profile(profile_name)
        
        # Select optimal strategy
        selection_result = enhanced_selector.select_optimal_strategy(data)
        
        # Print results
        logger.info(f"Market Regime: {selection_result['market_regime']}")
        logger.info(f"Trading Time Optimality: {selection_result['trading_time_optimality']}")
        logger.info(f"Should Trade: {selection_result['should_trade']}")
        
        logger.info("Selected Strategies:")
        for strategy in selection_result['strategies']:
            logger.info(f"  {strategy['strategy_type']}:")
            logger.info(f"    Total Score: {strategy['total_score']:.4f}")
            logger.info(f"    Regime Score: {strategy['regime_score']:.4f}")
            logger.info(f"    Risk Score: {strategy['risk_score']:.4f}")
            logger.info(f"    Allocation Weight: {strategy.get('allocation_weight', 0):.2f}")

def test_trading_time_optimization() -> None:
    """
    Test the trading time optimization functionality.
    """
    logger.info("\nTesting trading time optimization")
    
    # Initialize enhanced selector
    enhanced_selector = EnhancedStrategySelector()
    
    # Check different times
    test_times = [
        # Format: (day, hour, expected_optimality)
        ("Monday", 3, "GOOD"),        # Asian session
        ("Monday", 9, "OPTIMAL"),     # London open
        ("Monday", 14, "OPTIMAL"),    # London/NY overlap
        ("Monday", 20, "GOOD"),       # NY afternoon
        ("Friday", 22, "POOR"),       # Friday night
        ("Saturday", 12, "POOR"),     # Weekend
        ("Sunday", 22, "SUBOPTIMAL")  # Sunday pre-Asian
    ]
    
    for day_name, hour, expected in test_times:
        # Convert to datetime 
        day_index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)
        test_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=pd.Timestamp.now().weekday()) + pd.Timedelta(days=day_index)
        test_time = test_date.replace(hour=hour)
        
        # Evaluate
        optimality = enhanced_selector.evaluate_trading_time(test_time)
        should_trade, reason = enhanced_selector.should_trade_now(test_time)
        
        logger.info(f"{day_name} {hour:02d}:00 - Optimality: {optimality.name} (Expected: {expected})")
        logger.info(f"  Should Trade: {should_trade} - {reason}")

def generate_strategy_map() -> None:
    """
    Generate and display strategy map for regime/risk combinations.
    """
    logger.info("\nGenerating strategy map for market regimes and risk profiles")
    
    # Initialize with a conservative profile
    risk_manager = RiskProfileManager()
    risk_manager.create_profile("map_test", RiskToleranceLevel.MODERATE)
    
    enhanced_selector = EnhancedStrategySelector(risk_profile_manager=risk_manager)
    
    # Generate map
    strategy_map = enhanced_selector.export_strategy_map()
    
    # Display map
    logger.info("Strategy Map by Market Regime:")
    for regime, strategies in strategy_map.items():
        if not strategies:
            continue
            
        logger.info(f"\nRegime: {regime}")
        for i, strategy in enumerate(strategies[:3]):  # Show top 3 for each regime
            logger.info(f"  {i+1}. {strategy['strategy_type']} (Score: {strategy['compatibility_score']:.2f})")

def main():
    """Main test function."""
    logger.info("Starting Enhanced Strategy Selection tests")
    
    # Load or generate sample data
    data = load_sample_data()
    
    # Run tests
    test_strategy_selection_with_profiles(data)
    test_trading_time_optimization()
    generate_strategy_map()
    
    logger.info("\nTests completed successfully")

if __name__ == "__main__":
    main()
